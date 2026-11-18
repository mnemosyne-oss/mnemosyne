"""Release gate catches mismatched/unsafe tags before registry credentials are used."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'deploy/docker/validate_release.py'


@pytest.mark.parametrize('tag,version,stable', [('v4.0.0','4.0.0',True),('v4.0.1rc1','4.0.1rc1',False),('v4.1.0a1','4.1.0a1',False),('v4.1.0b2','4.1.0b2',False)])
def test_release_tag_matches_literal_source(tmp_path,tag,version,stable):
    source=tmp_path/'__init__.py'
    source.write_text(f'__version__ = "{version}"\nraise RuntimeError("must not execute")\n')
    result=subprocess.run([sys.executable,str(SCRIPT),tag,'--version-file',str(source)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)=={'version':version,'stable':stable}


@pytest.mark.parametrize('tag,source', [('v4.0.1','__version__="4.0.0"'),('main','__version__="4.0.0"'),('v0.7.7','__version__="4.0.0"'),('v4.0.0;echo bad','__version__="4.0.0"'),('v4.0.0','__version__=str(4)'),('v4.0.0','value=1'),('v4.0.0','__version__="4.0.0"\n__version__="4.0.0"')])
def test_release_gate_refuses_invalid_source_or_tag(tmp_path,tag,source):
    file=tmp_path/'__init__.py';file.write_text(source)
    assert subprocess.run([sys.executable,str(SCRIPT),tag,'--version-file',str(file)],capture_output=True).returncode != 0


def workflow_run(name):
    """Execute the real pre-login steps offline, without loading a YAML extra."""
    import re
    import textwrap
    workflow = (SCRIPT.parents[2] / '.github/workflows/docker-publish.yml').read_text()
    step = workflow.split('      - name: ' + name + '\n', 1)[1]
    body = step.split('        run: |\n', 1)[1]
    return textwrap.dedent(re.split(r'\n      - ', body, maxsplit=1)[0])


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True, stderr=subprocess.STDOUT).strip()


def release_repo(tmp_path):
    import shutil
    repo = tmp_path / 'repo'
    repo.mkdir()
    git(repo, 'init', '--initial-branch=main')
    git(repo, 'config', 'user.name', 'Release gate test')
    git(repo, 'config', 'user.email', 'release-gate@example.invalid')
    git(repo, 'config', 'core.hooksPath', '/dev/null')
    (repo / 'mnemosyne').mkdir()
    (repo / 'mnemosyne/__init__.py').write_text('__version__ = "4.0.0"\nraise RuntimeError("never import release code")\n')
    (repo / 'deploy/docker').mkdir(parents=True)
    shutil.copy(SCRIPT, repo / 'deploy/docker/validate_release.py')
    git(repo, 'add', '.')
    git(repo, 'commit', '-m', 'Release source')
    return repo


@pytest.mark.parametrize('kind', ['lightweight', 'annotated', 'mismatch', 'nonancestor'])
def test_workflow_resolves_only_matching_ancestor_release(tmp_path, kind):
    import os
    repo = release_repo(tmp_path)
    expected = git(repo, 'rev-parse', 'HEAD')
    tag = 'v4.0.1' if kind == 'mismatch' else 'v4.0.0'
    if kind == 'nonancestor':
        git(repo, 'checkout', '--orphan', 'other-history')
        git(repo, 'commit', '-m', 'Unrelated source')
    git(repo, 'tag', *(['-a', '-m', 'Annotated release'] if kind == 'annotated' else []), tag)
    if kind == 'nonancestor':
        git(repo, 'checkout', 'main')
    output = tmp_path / 'output'
    output.mkdir()
    env_file = tmp_path / 'github-env'
    env = dict(os.environ, RELEASE_TAG=tag, GITHUB_ENV=str(env_file),
               PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'])
    # Redirect only temporary outputs; run the workflow's actual shell/Python gate.
    script = workflow_run('Resolve trusted release source').replace('/tmp/', str(output) + '/')
    result = subprocess.run(['bash', '-eo', 'pipefail', '-c', script], cwd=repo, env=env,
                            capture_output=True, text=True, timeout=15)
    if kind in ('mismatch', 'nonancestor'):
        assert result.returncode != 0, result.stdout + result.stderr
        assert not env_file.exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert git(repo, 'rev-parse', 'HEAD') == expected
        lines = dict(line.split('=', 1) for line in env_file.read_text().splitlines())
        assert lines['RELEASE_SHA'] == expected
        assert lines['RELEASE_VERSION'] == '4.0.0'
        image = 'ghcr.io/mnemosyne-oss/mnemosyne-mcp'
        assert lines['IMAGE_TAGS'].split(',') == [image + ':4.0.0', image + ':sha-' + expected, image + ':latest']


@pytest.mark.parametrize('case,allowed', [('both', True), ('missing-arm', False), ('failed-arm', False),
                                         ('latest-failed', False), ('other-app', False)])
def test_workflow_requires_both_latest_native_checks(tmp_path, case, allowed):
    import os
    amd = {'id': 1, 'name': 'container (ubuntu-24.04)', 'conclusion': 'success', 'app': {'slug': 'github-actions'}}
    arm = {'id': 2, 'name': 'container (ubuntu-24.04-arm)', 'conclusion': 'success', 'app': {'slug': 'github-actions'}}
    checks = [amd, arm]
    if case == 'missing-arm':
        checks = [amd]
    elif case == 'failed-arm':
        arm['conclusion'] = 'failure'
    elif case == 'latest-failed':
        checks.append(dict(arm, id=3, conclusion='failure'))
    elif case == 'other-app':
        arm['app'] = {'slug': 'untrusted-app'}
    (tmp_path / 'checks.json').write_text(json.dumps({'check_runs': checks}))
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    gh = fake_bin / 'gh'
    gh.write_text('#!/bin/sh\n[ "$1" = api ] || exit 99\ncat "$CHECK_FIXTURE"\n')
    gh.chmod(0o755)
    env = dict(os.environ, PATH=str(fake_bin) + os.pathsep + str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'],
               GITHUB_REPOSITORY='example/offline', RELEASE_SHA='fixture-commit', CHECK_FIXTURE=str(tmp_path / 'checks.json'))
    script = workflow_run('Require successful native container checks on release source').replace('/tmp/', str(tmp_path) + '/')
    result = subprocess.run(['bash', '-eo', 'pipefail', '-c', script], env=env,
                            capture_output=True, text=True, timeout=15)
    assert (result.returncode == 0) == allowed, result.stdout + result.stderr
    if not allowed:
        assert 'Missing successful release-source check:' in result.stderr
