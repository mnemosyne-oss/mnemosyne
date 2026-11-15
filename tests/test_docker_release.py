"""Release gate catches mismatched/unsafe tags before registry credentials are used."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'deploy/docker/validate_release.py'


@pytest.mark.parametrize('tag,version,stable', [('v4.0.0','4.0.0',True),('v4.0.1rc1','4.0.1rc1',False)])
def test_release_tag_matches_literal_source(tmp_path,tag,version,stable):
    source=tmp_path/'__init__.py'
    source.write_text(f'__version__ = "{version}"\nraise RuntimeError("must not execute")\n')
    result=subprocess.run([sys.executable,str(SCRIPT),tag,'--version-file',str(source)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)=={'version':version,'stable':stable}


@pytest.mark.parametrize('tag,source', [('v4.0.1','__version__="4.0.0"'),('main','__version__="4.0.0"'),('v0.7.7','__version__="4.0.0"'),('v4.0.0;echo bad','__version__="4.0.0"'),('v4.0.0','__version__=str(4)')])
def test_release_gate_refuses_invalid_source_or_tag(tmp_path,tag,source):
    file=tmp_path/'__init__.py';file.write_text(source)
    assert subprocess.run([sys.executable,str(SCRIPT),tag,'--version-file',str(file)],capture_output=True).returncode != 0
