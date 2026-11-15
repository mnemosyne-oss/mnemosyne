"""Executable acceptance of image, publication boundaries and Compose merging."""
import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import pytest

httpx = pytest.importorskip("httpx")
httpx2 = pytest.importorskip("httpx2")
pytest.importorskip("mcp")
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

ROOT = Path(__file__).resolve().parents[2]


def docker(*args, **kwargs):
    return subprocess.run(['docker', *args], check=True, capture_output=True, text=True, **kwargs).stdout.strip()


@contextmanager
def volume():
    name = 'mnemosyne-pr-' + uuid.uuid4().hex
    docker('volume', 'create', name)
    try:
        yield name
    finally:
        docker('volume', 'rm', name)


@contextmanager
def service(image, data, cache):
    name = 'mnemosyne-pr-' + uuid.uuid4().hex
    docker('run', '-d', '--name', name, '-p', '127.0.0.1::8080', '-v', data+':/data', '-v', cache+':/cache',
           '-e', 'MNEMOSYNE_MCP_TOKENS={"hermes":"test-hermes","codex":"test-codex","claude-code":"test-claude"}',
           '-e', 'MNEMOSYNE_MCP_ALLOWED_HOSTS=127.0.0.1:*,localhost:*', image,
           '--transport', 'streamable-http', '--host', '0.0.0.0', '--port', '8080')
    port = docker('port', name, '8080/tcp').rsplit(':', 1)[1]
    url = f'http://127.0.0.1:{port}/mcp'
    try:
        for _ in range(120):
            try:
                if httpx.get(url, timeout=.5).status_code == 401:
                    break
            except httpx.TransportError:
                pass
            time.sleep(.5)
        else:
            pytest.fail('Container did not become reachable')
        yield name, url
    finally:
        docker('rm', '-f', name)


async def call(url, token, tool=None, arguments=None, ca=True):
    async with httpx2.AsyncClient(headers={'Authorization':f'Bearer {token}'}, timeout=180, verify=ca, trust_env=False) as client:
        async with streamable_http_client(url, http_client=client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                if tool is None:
                    assert (await session.list_tools()).tools
                    return
                result = await session.call_tool(tool, arguments or {})
                assert not result.is_error, result
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
    if overlay=='compose.caddy.yml':
        assert c['services']['caddy']['ports']


@pytest.mark.parametrize('key',['MNEMOSYNE_IMAGE','MNEMOSYNE_MCP_TOKENS','MNEMOSYNE_MCP_ALLOWED_HOSTS'])
def test_compose_requires_configuration(compose_cmd,key):
    assert render(compose_cmd,extra={key:''},check=False).returncode != 0


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
            assert httpx.get(url).status_code==401
            assert httpx.get(url,headers={'Authorization':'Bearer wrong'}).status_code==401
            assert httpx.get(url,headers={'Authorization':'Bearer test-hermes','Host':'bad.invalid'}).status_code==421
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


def test_caddy_internal_https_and_persistent_ca(image, tmp_path):
    import ssl
    caddy_image = 'caddy:2-alpine@sha256:003a97808e781aa9f7e75cc2f43d34d461099031faf5e20faefb9f61126836f0'
    for file in ['Caddyfile.lan', 'Caddyfile.public']:
        docker('run','-i','--rm','-e','MNEMOSYNE_HOSTNAME=localhost','--entrypoint','caddy',caddy_image,'validate','--config','-','--adapter','caddyfile', input=(ROOT/'deploy/docker'/file).read_text())
    network='mnemosyne-pr-'+uuid.uuid4().hex
    name='mnemosyne-pr-'+uuid.uuid4().hex
    docker('network','create',network)
    try:
        with volume() as data, volume() as cache, volume() as ca, volume() as config:
            with service(image,data,cache) as (upstream,_):
                docker('network','connect','--alias','mnemosyne-mcp',network,upstream)
                docker('create','--name',name,'--network',network,'-p','127.0.0.1::443',
                       '-e','MNEMOSYNE_HOSTNAME=localhost',
                       '-v',ca+':/data','-v',config+':/config',caddy_image)
                docker('cp',str(ROOT/'deploy/docker/Caddyfile.lan'),name+':/etc/caddy/Caddyfile')
                docker('start',name)
                try:
                    port=docker('port',name,'443/tcp').rsplit(':',1)[1]
                    url=f'https://localhost:{port}/mcp'
                    root='/data/caddy/pki/authorities/local/root.crt'
                    for _ in range(60):
                        try:
                            docker('cp',name+':'+root,str(tmp_path/'root.crt'))
                            break
                        except subprocess.CalledProcessError:
                            time.sleep(.5)
                    context=ssl.create_default_context(cafile=str(tmp_path/'root.crt'))
                    with pytest.raises(httpx.TransportError):
                        httpx.get(url,trust_env=False)
                    assert httpx.get(url,headers={'Authorization':'Bearer wrong'},verify=context,trust_env=False).status_code==401
                    asyncio.run(call(url,'test-hermes',ca=context))
                    stored=asyncio.run(call(url,'test-hermes','mnemosyne_remember',{'content':'Amber fox shares HTTPS memory'},ca=context))
                    assert stored['status']=='stored'
                    rows=asyncio.run(call(url,'test-codex','mnemosyne_recall',{'query':'Amber fox'},ca=context))['results']
                    assert any('Amber fox' in r['content'] for r in rows)
                    before=(tmp_path/'root.crt').read_bytes()
                    docker('rm','-f',name)
                    docker('create','--name',name,'--network',network,'-p','127.0.0.1::443',
                           '-e','MNEMOSYNE_HOSTNAME=localhost','-v',ca+':/data','-v',config+':/config',caddy_image)
                    docker('cp',str(ROOT/'deploy/docker/Caddyfile.lan'),name+':/etc/caddy/Caddyfile')
                    docker('start',name)
                    docker('cp',name+':'+root,str(tmp_path/'root-after.crt'))
                    assert (tmp_path/'root-after.crt').read_bytes()==before
                    new_port=docker('port',name,'443/tcp').rsplit(':',1)[1]
                    for _ in range(30):
                        try:
                            asyncio.run(call(f'https://localhost:{new_port}/mcp','test-codex',ca=context))
                            break
                        except Exception:
                            time.sleep(.2)
                    else:
                        pytest.fail('Recreated Caddy did not accept original CA')
                finally:
                    docker('rm','-f',name)
    finally:
        docker('network','rm',network)


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
    with volume() as data:
        docker('run','--rm','-u','0','-v',data+':/data','--entrypoint','sh',image,'-c','touch /data/root-owned && chown 0:0 /data && chmod 700 /data')
        result=subprocess.run(['docker','run','--rm','-v',data+':/data','--entrypoint','python',image,'-c','from pathlib import Path; Path("/data/probe").write_text("x")'],capture_output=True,text=True)
        assert result.returncode!=0
        assert 'PermissionError' in result.stderr


@pytest.mark.parametrize('tokens,hosts',[('', 'localhost:*'),('{"a":"x"}', ''),('not-json','localhost:*')])
def test_container_refuses_invalid_auth_configuration(image,tokens,hosts):
    result=subprocess.run(['docker','run','--rm','-e','MNEMOSYNE_MCP_TOKENS='+tokens,'-e','MNEMOSYNE_MCP_ALLOWED_HOSTS='+hosts,image,'--transport','streamable-http','--host','0.0.0.0'],capture_output=True,text=True,timeout=20)
    assert result.returncode != 0


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
