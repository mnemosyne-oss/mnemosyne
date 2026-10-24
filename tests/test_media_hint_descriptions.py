"""Public hint contracts, extracted without importing any provider surface."""

import ast
from pathlib import Path
import re

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SURFACES = (
    "mnemosyne/tool_schemas.py",
    "integrations/hermes/src/mnemosyne_hermes/tools.py",
    "hermes_memory_provider/__init__.py",
)


def _hint_description(relative_path):
    tree = ast.parse((_ROOT / relative_path).read_text(encoding="utf-8"))
    definitions = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "REMEMBER_MEDIA_SCHEMA"
                for target in node.targets)
    ]
    assert len(definitions) == 1, "Target the actual remember_media schema assignment"
    schema = ast.literal_eval(definitions[0])
    assert schema["name"] == "mnemosyne_remember_media"
    hint = schema["parameters"]["properties"]["hint"]
    assert hint["type"] == "string"
    assert isinstance(hint["description"], str)
    return hint["description"]


@pytest.mark.parametrize("surface", _SURFACES)
def test_remember_media_hint_documents_the_image_override_contract(surface):
    text = " ".join(_hint_description(surface).lower().split())
    # Assert semantic clauses, not an entire fixed English sentence. Selecting
    # the named schema and exact properties path excludes unrelated descriptions.
    assert "guidance" in text
    assert re.search(r"openai-compatible image adapter.*non-empty hint.*replaces.*entire prompt", text)
    assert re.search(r"not appended", text)
    assert re.search(r"caller must include.*expected json", text)
    assert "summary" in text and "moments" in text
    assert re.search(r"empty or omitted hint.*configured.*default.*template", text)


def test_all_hint_surfaces_have_matching_descriptions():
    descriptions = [_hint_description(surface) for surface in _SURFACES]
    assert all(description == descriptions[0] for description in descriptions)
