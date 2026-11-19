"""Check the effective embedding requirements for both packaging paths."""

import ast
from pathlib import Path

import pytest
from packaging.requirements import Requirement

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["pyproject.toml", "setup.py"])
def extras(request):
    source = (ROOT / request.param).read_text(encoding="utf-8")
    if request.param == "pyproject.toml":
        return tomllib.loads(source)["project"]["optional-dependencies"]
    setup_call = next(
        node.value
        for node in ast.parse(source).body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "setup"
    )
    return ast.literal_eval(
        next(kw.value for kw in setup_call.keywords if kw.arg == "extras_require")
    )


@pytest.mark.parametrize("extra", ["embeddings", "all"])
@pytest.mark.parametrize(
    ("platform", "machine", "python_version", "expected_ort"),
    [
        ("darwin", "x86_64", "3.10", ">=1.21.0,<1.24"),
        ("darwin", "x86_64", "3.13", ">=1.21.0,<1.24"),
        ("darwin", "x86_64", "3.14", None),
        ("darwin", "arm64", "3.14", ">=1.21.0"),
        ("linux", "x86_64", "3.11", ">=1.21.0"),
        ("linux", "aarch64", "3.14", ">=1.21.0"),
        ("win32", "AMD64", "3.14", ">=1.21.0"),
    ],
)
def test_effective_embedding_requirements(
    extras, extra, platform, machine, python_version, expected_ort
):
    """Keep the floor and the Intel wheel exception in both extras."""
    environment = {
        "sys_platform": platform,
        "platform_machine": machine,
        "python_version": python_version,
        "python_full_version": f"{python_version}.0",
        "extra": extra,
    }
    requirements = [Requirement(declaration) for declaration in extras[extra]]
    active = [
        req for req in requirements
        if req.marker is None or req.marker.evaluate(environment)
    ]
    ort = [req for req in active if req.name == "onnxruntime"]
    fastembed = [req for req in active if req.name == "fastembed"]
    assert any(req.name == "sqlite-vec" for req in active)
    if expected_ort is None:
        assert not ort
        assert not fastembed
    else:
        assert len(fastembed) == 1
        assert len(ort) == 1
        assert ort[0].specifier == Requirement(f"onnxruntime{expected_ort}").specifier
