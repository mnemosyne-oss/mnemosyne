"""The faster Hangul predicate must preserve lexical recall, not broaden it."""

import pytest

from mnemosyne.core import beam
from mnemosyne.core.beam import BeamMemory


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", False),
        ("configuration", False),
        ("конфигурация", False),
        ("東京 数据库", False),
        ("\uabff", False),
        ("\uac00", True),
        ("\ud7a3", True),
        ("\ud7a4", False),
        ("\ud7af", False),
        ("\u1100\u1161", False),  # Decomposed 가: no normalization here.
        ("\u3131\u314f", False),  # Compatibility Jamo.
        ("\ua960\ud7b0", False),  # Extended Jamo.
        ("\ud800\udfff\U0010ffff", False),
        ("prefix가", True),
        ("힣suffix", True),
        ("ModelForge가 설정", True),
        ("\uabff\ud7a4\uac00\ud7af", True),
    ],
)
def test_has_hangul_uses_only_precomposed_syllables(text, expected):
    assert beam._has_hangul(text) is expected


@pytest.mark.parametrize("layer", ["working", "episodic"])
@pytest.mark.parametrize(
    "query, contents",
    [
        (
            "백업을 보관 기간",
            ["백업 보관 기간", "백업을 보관한다 기간은", "보관함", "unrelated notes"],
        ),
        (
            "ModelForge가 설정 파일",
            [
                "ModelForge 설정 파일",
                "ModelForge가 설정한다 파일",
                "ModelForgeXYZ 설정",
                "unrelated notes",
            ],
        ),
        (
            "gateway telemetry",
            [
                "gateway telemetry",
                "gateway telemetry notes",
                "gatewayish telemetryish",
                "unrelated notes",
            ],
        ),
    ],
)
def test_public_recall_preserves_order_scores_and_excludes_distractors(
    tmp_path, monkeypatch, layer, query, contents
):
    memory = BeamMemory(session_id="issue-1192", db_path=tmp_path / "memory.db")
    # The two predicate variants must see identical recency, not elapsed time.
    monkeypatch.setattr(beam, "_recency_decay", lambda timestamp: 1.0)
    ids = []
    for index, content in enumerate(contents):
        importance = 0.8 - index * 0.1
        if layer == "working":
            ids.append(memory.remember(content, source="test", importance=importance))
        else:
            row_id = f"row-{index}"
            conn = beam._get_connection(memory.db_path)
            conn.execute(
                "INSERT INTO episodic_memory"
                " (id, content, source, session_id, importance)"
                " VALUES (?, ?, 'test', 'issue-1192', ?)",
                (row_id, content, importance),
            )
            conn.commit()
            ids.append(row_id)

    current = memory.recall(query, top_k=10)
    monkeypatch.setattr(
        beam, "_has_hangul", lambda text: any("가" <= ch <= "힣" for ch in text)
    )
    reference = memory.recall(query, top_k=10)

    assert [row["id"] for row in current] == ids[:2]
    assert [row["id"] for row in reference] == ids[:2]
    assert [(row["id"], row["score"]) for row in current] == [
        (row["id"], row["score"]) for row in reference
    ]
