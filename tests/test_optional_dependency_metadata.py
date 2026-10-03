"""Regression coverage for optional dependency constraints."""

from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ONNXRUNTIME_REQUIREMENT = "onnxruntime>=1.21.0,<1.29"
EXPECTED_EXTRAS = {"embeddings", "all"}


def _pyproject_optional_dependencies() -> dict[str, list[str]]:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    optional_dependencies = pyproject.split("[project.optional-dependencies]", 1)[1].split(
        "\n[", 1
    )[0]
    assignments = re.findall(
        r"^(embeddings|all)\s*=\s*(\[[^\n]+\])$", optional_dependencies, re.MULTILINE
    )
    dependencies = {
        extra: ast.literal_eval(requirements) for extra, requirements in assignments
    }
    assert set(dependencies) == EXPECTED_EXTRAS
    assert optional_dependencies.count(ONNXRUNTIME_REQUIREMENT) == len(EXPECTED_EXTRAS)
    return dependencies


def _setup_py_optional_dependencies() -> dict[str, list[str]]:
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    setup_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ((isinstance(node.func, ast.Name) and node.func.id == "setup")
             or (isinstance(node.func, ast.Attribute) and node.func.attr == "setup"))
    )
    extras_keyword = next(
        keyword for keyword in setup_call.keywords if keyword.arg == "extras_require"
    )
    return ast.literal_eval(extras_keyword.value)


def test_embedding_extras_bound_onnxruntime_below_1_29():
    """Avoid onnxruntime 1.29, whose import invokes unavailable ``blkid``."""
    for optional_dependencies in (
        _pyproject_optional_dependencies(),
        _setup_py_optional_dependencies(),
    ):
        for extra in EXPECTED_EXTRAS:
            assert optional_dependencies[extra].count(ONNXRUNTIME_REQUIREMENT) == 1

        extras_with_requirement = {
            extra
            for extra, requirements in optional_dependencies.items()
            if ONNXRUNTIME_REQUIREMENT in requirements
        }
        assert extras_with_requirement == EXPECTED_EXTRAS
