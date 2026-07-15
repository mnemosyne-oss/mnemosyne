"""Regression tests for source-distribution package discovery."""

from pathlib import Path
import ast
import tomllib


ROOT = Path(__file__).parents[1]
REQUIRED_EXCLUDES = {"integrations*", "examples*", "build*", "dist*"}


def test_pyproject_excludes_non_distribution_namespaces():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    excludes = set(data["tool"]["setuptools"]["packages"]["find"]["exclude"])

    assert REQUIRED_EXCLUDES <= excludes


def test_legacy_setup_uses_matching_package_exclusions():
    tree = ast.parse((ROOT / "setup.py").read_text())
    find_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "find_packages"
    )
    exclude_keyword = next(
        (keyword for keyword in find_call.keywords if keyword.arg == "exclude"),
        None,
    )

    assert exclude_keyword is not None
    assert REQUIRED_EXCLUDES <= set(ast.literal_eval(exclude_keyword.value))
