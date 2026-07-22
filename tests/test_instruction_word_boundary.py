from mnemosyne.core.beam import BeamMemory


def test_whenever_is_not_extracted_as_never(tmp_path):
    beam = BeamMemory(session_id="instruction-boundary", db_path=tmp_path / "test.db")

    counts = beam.extract_and_store_facts(
        "Good - whenever needed we can use it. I've disabled it.", message_idx=1
    )

    instructions = beam.conn.execute(
        "SELECT instruction, topic FROM memoria_instructions"
    ).fetchall()
    assert counts.get("instruction", 0) == 0
    assert instructions == []


def test_never_at_word_boundary_remains_an_instruction(tmp_path):
    beam = BeamMemory(session_id="instruction-boundary", db_path=tmp_path / "test.db")

    counts = beam.extract_and_store_facts(
        "Never discard completed work without checking it.", message_idx=1
    )

    instruction = beam.conn.execute(
        "SELECT instruction FROM memoria_instructions"
    ).fetchone()
    assert counts["instruction"] == 1
    assert instruction[0] == "Never discard completed work without checking it"
