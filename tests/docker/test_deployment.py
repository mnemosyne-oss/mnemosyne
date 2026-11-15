"""Executable acceptance of image, publication boundaries and Compose merging."""
import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
import warnings

import pytest

httpx = pytest.importorskip("httpx")
httpx2 = pytest.importorskip("httpx2")
pytest.importorskip("mcp")
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

ROOT = Path(__file__).resolve().parents[2]


def wait_for_https(url, context, timeout=15):
    deadline = time.monotonic() + timeout
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            response = httpx.get(url, headers={'Authorization': 'Bearer wrong'},
                                 verify=context, trust_env=False, timeout=min(1, remaining))
        except httpx.TransportError:
            time.sleep(min(.1, max(0, deadline - time.monotonic())))
        else:
            assert response.status_code == 401, response.text
            return response
    pytest.fail('HTTPS did not become reachable with the configured CA')


class DockerCommandError(subprocess.CalledProcessError):
    def __str__(self):
        return super().__str__() + "\n" + (self.stderr or self.output or '')


def command(cmd, **kwargs):
    kwargs.setdefault('timeout', 180)
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode:
        raise DockerCommandError(result.returncode, result.args, result.stdout, result.stderr)
    return result.stdout.strip()


def docker(*args, **kwargs):
    return command(['docker', *args], **kwargs)


def cleanup_command(cmd, **kwargs):
    # A secondary cleanup failure must not replace the primary test failure.
    import sys
    failed = sys.exc_info()[0] is not None
    try:
        command(cmd, **kwargs)
    except (subprocess.SubprocessError, OSError) as exc:
        if not failed:
            raise
        warnings.warn(f'Docker cleanup failed: {exc}', RuntimeWarning, stacklevel=2)


def cleanup(*args):
    cleanup_command(['docker', *args])


@contextmanager
def volume():
    name = 'mnemosyne-pr-' + uuid.uuid4().hex
    docker('volume', 'create', name)
    try:
        yield name
    finally:
        cleanup('volume', 'rm', name)


@contextmanager
def service(image, data, cache, hosts='127.0.0.1:*,localhost:*'):
    name = 'mnemosyne-pr-' + uuid.uuid4().hex
    docker('run', '-d', '--name', name, '-p', '127.0.0.1::8080', '-v', data+':/data', '-v', cache+':/cache',
           '-e', 'MNEMOSYNE_MCP_TOKENS={"hermes":"test-hermes","codex":"test-codex","claude-code":"test-claude"}',
           '-e', 'MNEMOSYNE_MCP_ALLOWED_HOSTS='+hosts, image,
           '--transport', 'streamable-http', '--host', '0.0.0.0', '--port', '8080')
    try:
        port = docker('port', name, '8080/tcp').rsplit(':', 1)[1]
        url = f'http://127.0.0.1:{port}/mcp'
        for _ in range(120):
            try:
                if httpx.get(url, timeout=.5, trust_env=False).status_code == 401:
                    break
            except httpx.TransportError:
                pass
            time.sleep(.5)
        else:
            pytest.fail('Container did not become reachable:\n' + docker('logs', name))
        yield name, url
    finally:
        cleanup('rm', '-f', name)


async def call(url, token, tool=None, arguments=None, ca=True, expect_error=False):
    async with httpx2.AsyncClient(headers={'Authorization':f'Bearer {token}'}, timeout=180, verify=ca, trust_env=False) as client:
        async with streamable_http_client(url, http_client=client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                if tool is None:
                    assert (await session.list_tools()).tools
                    return
                result = await session.call_tool(tool, arguments or {})
                assert result.is_error == expect_error, result
                return json.loads(result.content[0].text)


def render(compose_cmd, *overrides, extra=None, check=True):
    env = dict(os.environ, MNEMOSYNE_IMAGE='mnemosyne-mcp:pr',
               MNEMOSYNE_MCP_TOKENS='{"hermes":"test-$-quote\\\""}',
               MNEMOSYNE_MCP_ALLOWED_HOSTS='mnemosyne.home.arpa',
               MNEMOSYNE_BIND_IP='192.0.2.10', MNEMOSYNE_PROXY_NETWORK='test-proxy',
               MNEMOSYNE_HOSTNAME='mnemosyne.home.arpa', MNEMOSYNE_CADDY_BIND_IP='192.0.2.10',
               MNEMOSYNE_CADDYFILE='./deploy/docker/Caddyfile.lan')
    env.update(extra or {})
    cmd = compose_cmd + ['-p','mnemosyne-pr-config','-f',str(ROOT/'docker-compose.yml')]
    for file in overrides:
        cmd += ['-f',str(ROOT/'deploy/docker'/file)]
    p = subprocess.run(cmd+['config','--format','json'],env=env,capture_output=True,text=True)
    if not check:
        return p
    assert p.returncode == 0, p.stderr
    config=json.loads(p.stdout)
    assert config['services']['mnemosyne-mcp']['environment']['MNEMOSYNE_MCP_TOKENS'].replace('$$', '$') == env['MNEMOSYNE_MCP_TOKENS']
    return config


@pytest.mark.parametrize('overlay,host', [(None,'127.0.0.1'),('compose.lan.yml','192.0.2.10'),('compose.proxy.yml',None),('compose.caddy.yml',None)])
def test_compose_port_exposure(compose_cmd, overlay, host):
    c=render(compose_cmd,*([overlay] if overlay else []))
    ports=c['services']['mnemosyne-mcp'].get('ports',[])
    assert len(ports) == (1 if host else 0)
    if host:
        assert ports[0]['host_ip'] == host
        assert int(ports[0]['published']) == 8080
        assert ports[0]['target'] == 8080
    if overlay=='compose.caddy.yml':
        edge=c['services']['caddy']
        assert {(p['host_ip'], int(p['published']), p['target']) for p in edge['ports']} == {
            ('192.0.2.10',80,80), ('192.0.2.10',443,443)}
        mounts={v['target']:v for v in edge['volumes']}
        assert mounts['/data']['source']=='caddy-data'
        assert mounts['/config']['source']=='caddy-config'
        assert mounts['/etc/caddy/Caddyfile']['read_only']
        assert Path(mounts['/etc/caddy/Caddyfile']['source']) == ROOT/'deploy/docker/Caddyfile.lan'
    if overlay=='compose.proxy.yml':
        networks=c['services']['mnemosyne-mcp']['networks']
        assert set(networks)=={'default','proxy'}
        assert c['networks']['proxy']['external']
        assert c['networks']['proxy']['name']=='test-proxy'


@pytest.mark.parametrize('overlay,key', [
    (None,'MNEMOSYNE_IMAGE'), (None,'MNEMOSYNE_MCP_TOKENS'), (None,'MNEMOSYNE_MCP_ALLOWED_HOSTS'),
    ('compose.lan.yml','MNEMOSYNE_BIND_IP'), ('compose.proxy.yml','MNEMOSYNE_PROXY_NETWORK'),
    ('compose.caddy.yml','MNEMOSYNE_CADDY_BIND_IP'), ('compose.caddy.yml','MNEMOSYNE_CADDYFILE'),
    ('compose.caddy.yml','MNEMOSYNE_HOSTNAME')])
def test_compose_requires_configuration(compose_cmd,overlay,key):
    result=render(compose_cmd,*([overlay] if overlay else []),extra={key:''},check=False)
    assert result.returncode != 0
    assert key in result.stderr



def test_example_env_cannot_start_with_known_tokens(compose_cmd):
    env={k:v for k,v in os.environ.items() if not k.startswith('MNEMOSYNE_')}
    result=subprocess.run(compose_cmd+['--env-file',str(ROOT/'deploy/docker/.env.example'),
                                      '-f',str(ROOT/'docker-compose.yml'),'config'],
                          env=env,capture_output=True,text=True,timeout=20)
    assert result.returncode != 0
    assert 'MNEMOSYNE_MCP_TOKENS' in result.stderr


def test_documented_restore_override_is_isolated(compose_cmd,tmp_path):
    guide=(ROOT/'deploy/docker/README.md').read_text()
    yaml=guide.split("cat > backup/compose.restore.yml <<'YAML'\n",1)[1].split('\nYAML',1)[0]
    override=tmp_path/'restore.yml'
    override.write_text(yaml)
    config=render(compose_cmd,str(override))
    service=config['services']['mnemosyne-mcp']
    assert [(p['host_ip'],int(p['published']),p['target']) for p in service['ports']]==[('127.0.0.1',18080,8080)]
    assert service['environment']['MNEMOSYNE_MCP_ALLOWED_HOSTS']=='127.0.0.1:18080'

def test_image_nonroot_version_and_volumes(image):
    assert docker('run','--rm','--entrypoint','id',image,'-u') == '10001'
    expected = __import__('mnemosyne').__version__
    assert docker('run','--rm','--entrypoint','python',image,'-c','from importlib.metadata import version; print(version("mnemosyne-memory"))') == expected
    with volume() as data, volume() as cache:
        docker('run','--rm','-v',data+':/data','-v',cache+':/cache','--entrypoint','python',image,'-c',
               'from pathlib import Path; Path("/data/probe").write_text("ok"); Path("/cache/probe").write_text("ok")')


def test_shared_memory_recreation_and_stopped_backup(image):
    with volume() as data, volume() as cache, volume() as backup, volume() as restored:
        with service(image,data,cache) as (name,url):
            assert httpx.get(url,trust_env=False).status_code==401
            assert httpx.get(url,headers={'Authorization':'Bearer wrong'},trust_env=False).status_code==401
            assert httpx.get(url,headers={'Authorization':'Bearer test-hermes','Host':'bad.invalid'},trust_env=False).status_code==421
            docker('exec',name,'python','/opt/mnemosyne/healthcheck.py')
            docker('exec',name,'python','/opt/mnemosyne/readiness.py')
            result=asyncio.run(call(url,'test-hermes','mnemosyne_remember',{'content':'Copper otter prefers vanilla tea'}))
            assert result['status']=='stored'
            # Prove actual local model inference, not lexical fallback.
            docker('exec',name,'python','-c','from mnemosyne.core.embeddings import embed_query; v=embed_query("local embedding acceptance"); assert v is not None and len(v)>0')
            rows=asyncio.run(call(url,'test-codex','mnemosyne_recall',{'query':'Copper otter'}))['results']
            assert any('Copper otter' in r['content'] and r['author_id']=='hermes' for r in rows)
            assert docker('exec',name,'sh','-c','find /cache/fastembed -type f | head -1')
            docker('network','disconnect','bridge',name)
            docker('exec',name,'python','-c','from mnemosyne.core.embeddings import embed_query; v=embed_query("offline warm cache"); assert v is not None and len(v)>0')
            docker('stop','--time','10',name)
            assert docker('inspect','--format','{{.State.ExitCode}}',name) in {'0','143'}
        # No writer is running while making the backup. Preserve complete volume.
        docker('run','--rm','-u','0','-v',data+':/source:ro','-v',backup+':/backup','--entrypoint','sh',image,'-c','tar -C /source -cpf /backup/data.tar .')
        docker('run','--rm','-u','0','-v',restored+':/restore','-v',backup+':/backup:ro','--entrypoint','sh',image,'-c','tar -C /restore -xpf /backup/data.tar')
        for saved in [data,restored]:
            with service(image,saved,cache) as (_,url):
                rows=asyncio.run(call(url,'test-codex','mnemosyne_recall',{'query':'Copper otter'}))['results']
                assert any('Copper otter' in r['content'] for r in rows)


def test_caddy_internal_https_and_persistent_ca(image, compose_cmd, tmp_path):
    import ssl
    caddy_image=render(compose_cmd,'compose.caddy.yml')['services']['caddy']['image']
    for file in ['Caddyfile.lan','Caddyfile.public']:
        docker('run','-i','--rm','-e','MNEMOSYNE_HOSTNAME=localhost','--entrypoint','caddy',caddy_image,
               'validate','--config','-','--adapter','caddyfile',input=(ROOT/'deploy/docker'/file).read_text())
    # Use the actual deployment stack and persistent mounts. Only replace the
    # edge's host ports to avoid privileged/colliding 80/443 during acceptance.
    override=tmp_path/'compose.test.yml'
    override.write_text('services:\n  caddy:\n    ports: !override ["127.0.0.1::443"]\n')
    env=dict(os.environ,MNEMOSYNE_IMAGE=image,
             MNEMOSYNE_MCP_TOKENS='{"hermes":"test-hermes","codex":"test-codex"}',
             MNEMOSYNE_MCP_ALLOWED_HOSTS='localhost:*',MNEMOSYNE_HOSTNAME='localhost',
             MNEMOSYNE_CADDY_BIND_IP='127.0.0.1',MNEMOSYNE_CADDYFILE=str(ROOT/'deploy/docker/Caddyfile.lan'))
    cmd=compose_cmd+['-p','mnemosyne-pr-'+uuid.uuid4().hex,'-f',str(ROOT/'docker-compose.yml'),
                     '-f',str(ROOT/'deploy/docker/compose.caddy.yml'),'-f',str(override)]
    def compose(*args):
        return command(cmd+list(args),env=env)
    try:
        compose('up','-d','--no-build','--wait','--wait-timeout','90')
        name=compose('ps','-q','caddy')
        port=docker('port',name,'443/tcp').rsplit(':',1)[1]
        url=f'https://localhost:{port}/mcp'
        root='/data/caddy/pki/authorities/local/root.crt'
        for _ in range(60):
            try:
                docker('cp',name+':'+root,str(tmp_path/'root.crt'))
                break
            except subprocess.CalledProcessError:
                time.sleep(.5)
        else:
            pytest.fail('Caddy did not create its root CA:\n'+docker('logs',name))
        context=ssl.create_default_context(cafile=str(tmp_path/'root.crt'))
        wait_for_https(url,context)
        with pytest.raises(httpx.ConnectError,match='CERTIFICATE_VERIFY_FAILED'):
            httpx.get(url,trust_env=False)
        asyncio.run(call(url,'test-hermes',ca=context))
        stored=asyncio.run(call(url,'test-hermes','mnemosyne_remember',{'content':'Amber fox shares HTTPS memory'},ca=context))
        assert stored['status']=='stored'
        rows=asyncio.run(call(url,'test-codex','mnemosyne_recall',{'query':'Amber fox'},ca=context))['results']
        assert any('Amber fox' in r['content'] and r['author_id']=='hermes' for r in rows)
        before=(tmp_path/'root.crt').read_bytes()
        compose('up','-d','--no-deps','--no-build','--force-recreate','caddy')
        name=compose('ps','-q','caddy')
        docker('cp',name+':'+root,str(tmp_path/'root-after.crt'))
        assert (tmp_path/'root-after.crt').read_bytes()==before
        new_port=docker('port',name,'443/tcp').rsplit(':',1)[1]
        new_url=f'https://localhost:{new_port}/mcp'
        wait_for_https(new_url,context)
        asyncio.run(call(new_url,'test-codex',ca=context))
    finally:
        cleanup_command(cmd+['down','-v','--remove-orphans'],env=env)


def test_stdio_compatibility(image):
    from mcp import StdioServerParameters
    from mcp.client.stdio import stdio_client
    async def exercise():
        params=StdioServerParameters(command='docker',args=['run','-i','--rm','--no-healthcheck',image])
        async with stdio_client(params) as (read,write):
            async with ClientSession(read,write) as session:
                await session.initialize()
                assert (await session.list_tools()).tools
    asyncio.run(exercise())


def test_nonroot_unwritable_mount_fails(image):
    with volume() as data, volume() as cache:
        docker('run','--rm','-u','0','-v',data+':/data','--entrypoint','sh',image,'-c','touch /data/root-owned && chown 0:0 /data && chmod 700 /data')
        # MCP storage is lazy: exercise the default service's real write path,
        # rather than inventing a startup-refusal contract or overriding Python.
        with service(image,data,cache) as (_,url):
            result=asyncio.run(call(url,'test-hermes','mnemosyne_remember',
                                   {'content':'Unwritable mount must reject this'},expect_error=True))
            assert result['status']=='error'
            assert 'Permission denied' in result['message'] or 'PermissionError' in result['message']


@pytest.mark.parametrize('tokens,hosts,message',[
    ('', 'localhost:*', 'is set but empty'),
    ('{"a":"x"}', '', 'without a Host policy'),
    ('not-json','localhost:*','is not valid JSON'),
    ('{}','localhost:*','is empty; add at least one'),
    ('{"a":""}','localhost:*','contains an empty name or secret'),
    ('{"a":123}','localhost:*','entries must be JSON strings')])
def test_container_refuses_invalid_auth_configuration(image,tokens,hosts,message):
    name='mnemosyne-pr-'+uuid.uuid4().hex
    try:
        result=subprocess.run(['docker','run','--rm','--name',name,'--entrypoint','python',
                               '-e','MNEMOSYNE_MCP_TOKENS='+tokens,'-e','MNEMOSYNE_MCP_ALLOWED_HOSTS='+hosts,
                               image,'-m','mnemosyne.mcp_server','--transport','streamable-http','--host','0.0.0.0'],
                              capture_output=True,text=True,timeout=20)
        assert result.returncode != 0
        assert message in result.stderr, result.stderr
    finally:
        # --rm may already have removed it; also clean up if the client timed out.
        subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=20)



def test_default_cli_refuses_invalid_auth_without_traceback(image):
    name='mnemosyne-pr-'+uuid.uuid4().hex
    try:
        result=subprocess.run(['docker','run','--rm','--name',name,'-e','MNEMOSYNE_MCP_TOKENS={}',
                               '-e','MNEMOSYNE_MCP_ALLOWED_HOSTS=localhost:*',image,
                               '--transport','streamable-http','--host','0.0.0.0'],
                              capture_output=True,text=True,timeout=20)
        assert result.returncode==1
        assert result.stderr.rstrip().endswith('Error: cli_unexpected_failure')
        assert 'Traceback' not in result.stderr
    finally:
        subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=20)


def test_star_host_is_not_a_hostname_wildcard(image):
    with volume() as data, volume() as cache, service(image,data,cache,hosts='*') as (_,url):
        response=httpx.get(url,headers={'Authorization':'Bearer test-hermes'},trust_env=False)
        assert response.status_code==421


def test_compose_token_environment_roundtrip(compose_cmd,image):
    token='{"hermes":"test-$-quote\\\""}'
    env=dict(os.environ,MNEMOSYNE_IMAGE=image,MNEMOSYNE_MCP_TOKENS=token,MNEMOSYNE_MCP_ALLOWED_HOSTS='localhost:*')
    cmd=compose_cmd+['-p','mnemosyne-pr-'+uuid.uuid4().hex,'-f',str(ROOT/'docker-compose.yml')]
    try:
        result=subprocess.run(cmd+['run','--rm','--no-deps','--entrypoint','python','-e','EXPECTED='+token,'mnemosyne-mcp','-c','import os; assert os.environ["MNEMOSYNE_MCP_TOKENS"]==os.environ["EXPECTED"]'],env=env,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
    finally:
        subprocess.run(cmd+['down','-v'],env=env,check=True,capture_output=True)



def test_cold_cache_offline_requires_model_readiness(image):
    with volume() as cache:
        result=subprocess.run(['docker','run','--rm','--network','none',
                               '-e','HF_HUB_OFFLINE=1','-v',cache+':/cache',
                               '--entrypoint','python',image,'/opt/mnemosyne/readiness.py'],
                              capture_output=True,text=True,timeout=130)
        assert result.returncode == 1
        assert 'not ready' in result.stderr


def test_root_owned_legacy_volume_migration_preserves_memory(image):
    """Old root-run images leave data inaccessible to the new runtime UID."""
    with volume() as data, volume() as cache:
        mounts=['-v',data+':/data','-v',cache+':/cache']
        seed='''
from pathlib import Path
from mnemosyne.core.memory import Mnemosyne
memory = Mnemosyne(session_id="mcp_default", author_id="legacy-root")
assert memory.remember("Legacy copper otter keeps existing memories", scope="global")
Path("/cache/legacy-marker").write_text("keep this cache marker")
'''
        docker('run','--rm','--user','0',*mounts,'-e','MNEMOSYNE_NO_EMBEDDINGS=1',
               '--entrypoint','python',image,'-c',seed)
        docker('run','--rm','--user','0',*mounts,'--entrypoint','sh',image,'-ec',
               'chown -hR 0:0 /data /cache; chmod 700 /data /cache')
        denied=subprocess.run(['docker','run','--rm',*mounts,'--entrypoint','python',image,
                               '-c','from pathlib import Path; Path("/cache/legacy-marker").read_text()'],
                              capture_output=True,text=True)
        assert denied.returncode != 0
        assert 'PermissionError' in denied.stderr
        # Same scoped operation documented for stopped, dedicated named volumes.
        docker('run','--rm','--user','0',*mounts,'--entrypoint','sh',image,'-ec',
               'chown -hR 10001:10001 /data /cache')
        verify='''
import os
from pathlib import Path
from mnemosyne.core.memory import Mnemosyne
assert os.getuid() == 10001
assert os.stat("/data").st_uid == 10001
assert os.stat("/cache/legacy-marker").st_uid == 10001
assert Path("/cache/legacy-marker").read_text() == "keep this cache marker"
rows = Mnemosyne(session_id="mcp_default").recall("Legacy copper otter")
assert any(row["content"] == "Legacy copper otter keeps existing memories" and row["author_id"] == "legacy-root" for row in rows)
'''
        docker('run','--rm',*mounts,'-e','MNEMOSYNE_NO_EMBEDDINGS=1',
               '--entrypoint','python',image,'-c',verify)


def test_https_readiness_handles_early_closed_handshake(tmp_path):
    """A written CA does not imply that the TLS listener serves requests yet."""
    import datetime
    import ipaddress
    import socket
    import ssl
    import threading
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key=ec.generate_private_key(ec.SECP256R1())
    name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'localhost')])
    now=datetime.datetime.now(datetime.timezone.utc)
    cert=(x509.CertificateBuilder().subject_name(name).issuer_name(name)
          .public_key(key.public_key()).serial_number(x509.random_serial_number())
          .not_valid_before(now-datetime.timedelta(minutes=1))
          .not_valid_after(now+datetime.timedelta(hours=1))
          .add_extension(x509.BasicConstraints(ca=True,path_length=None),critical=True)
          .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]),critical=False)
          .sign(key,hashes.SHA256()))
    (tmp_path/'cert.pem').write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (tmp_path/'key.pem').write_bytes(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
    server_context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(str(tmp_path/'cert.pem'),str(tmp_path/'key.pem'))
    client_context=ssl.create_default_context(cafile=str(tmp_path/'cert.pem'))
    errors=[]
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));listener.listen();listener.settimeout(3)
        port=listener.getsockname()[1]
        def serve():
            try:
                early,_=listener.accept();early.close()
                connection,_=listener.accept()
                with server_context.wrap_socket(connection,server_side=True) as tls:
                    tls.recv(8192)
                    tls.sendall(b'HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
            except Exception as exc:errors.append(exc)
        thread=threading.Thread(target=serve,daemon=True);thread.start()
        try:
            response=wait_for_https(f'https://127.0.0.1:{port}/mcp',client_context,timeout=2)
            assert response.status_code==401
        finally:thread.join(timeout=4)
    assert not errors,errors
