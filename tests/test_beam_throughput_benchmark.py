"""Contract tests for the advisory harness; deliberately no timing thresholds."""
import copy
import json
import os
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import pytest

from _benchmarks import bench_beam_throughput as bench


def test_summary_uses_nearest_rank_and_whole_round_wall_time():
    """Quantiles use nearest ranks and QPS includes the entire round duration."""
    batches = [[i * 1_000_000 for i in range(1, 11)],
               [i * 1_000_000 for i in range(11, 21)]]
    result = bench.summarize(batches, [100_000_000, 200_000_000])
    assert result["calls"] == 20
    assert result["latency_ms"] == {
        "mean": 10.5, "p50": 10, "p95": 19, "min": 1, "max": 20,
    }
    assert result["round_qps"] == [100, 50]
    assert result["median_qps"] == 75


def test_measure_excludes_warmup_and_cycles_fixed_queries(monkeypatch):
    """Warmup calls are excluded while measured rounds retain fixed query order."""
    calls = []

    class FakeBeam:
        def recall(self, query, **kwargs):
            """Record invocation arguments and return a nonempty synthetic result."""
            calls.append((query, kwargs))
            return [{"id": "synthetic"}]

    clock = iter(range(0, 1000, 10))
    monkeypatch.setattr(bench.time, "perf_counter_ns", lambda: next(clock))
    args = SimpleNamespace(warmup_cycles=2, rounds=2, cycles_per_round=1, top_k=3)
    samples, counts, elapsed = bench.measure(FakeBeam(), args)
    assert [q for q, _ in calls] == list(bench.QUERIES) * 4
    assert all(kwargs == {"top_k": 3, **bench.RECALL_KWARGS} for _, kwargs in calls)
    assert samples == [[10] * 8] * 2
    assert counts == [[1] * 8] * 2
    assert elapsed == [170, 170]


def test_empty_recall_refuses_to_publish_timing():
    """An empty result must fail instead of appearing as fast useful recall."""
    beam = SimpleNamespace(recall=lambda *args, **kwargs: [])
    args = SimpleNamespace(warmup_cycles=1, rounds=1, cycles_per_round=1, top_k=3)
    with pytest.raises(RuntimeError, match="Empty recall"):
        bench.measure(beam, args)


def test_seed_has_fixed_corpus_and_no_ingestion_side_effects():
    """Independent databases receive identical rows through SQL-only seeding."""
    snapshots = []
    for _ in range(2):
        conn = sqlite3.connect(":memory:")
        try:
            for table in ("working_memory", "episodic_memory"):
                conn.execute(f"CREATE TABLE {table} (id, content, source, timestamp, "
                             "session_id, importance, veracity)")
            bench.seed(SimpleNamespace(conn=conn), 16)
            rows = {
                table: conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
                for table in ("working_memory", "episodic_memory")
            }
            assert all(len(batch) == 16 for batch in rows.values())
            assert rows["working_memory"][0] == (
                "working_memory-00000000",
                "zephyr0 calibration ledger item 00000000 pressure valve inspection record for working_memory",
                "benchmark", "2026-01-01T12:00:00", "throughput", 0.4, "stated",
            )
            snapshots.append(rows)
        finally:
            conn.close()
    assert snapshots[0] == snapshots[1]


@pytest.mark.parametrize("option", [
    "--rows-per-tier", "--top-k", "--warmup-cycles", "--cycles-per-round", "--rounds", "--timeout",
])
@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_cli_rejects_invalid_counts(option, value):
    """All workload counts and the timeout reject nonpositive or invalid input."""
    with pytest.raises(SystemExit) as exc:
        bench.parser().parse_args(["--machine", "test", option, value])
    assert exc.value.code == 2


def test_real_cli_isolated_linear_report_and_baseline(tmp_path):
    """Real worker runs preserve user data, exclude secrets and compare reports."""
    ambient = tmp_path / "user-data"
    ambient.mkdir()
    config = ambient / "config.yaml"
    config.write_text("polyphonic_recall: true\nrecall_content_cap: 1\n", encoding="utf-8")
    output = tmp_path / "reports" / "baseline.json"
    command = [
        sys.executable, str(bench.ROOT / "_benchmarks" / "bench_beam_throughput.py"),
        "--machine", "pytest-machine", "--rows-per-tier", "16",
        "--warmup-cycles", "1", "--rounds", "2", "--cycles-per-round", "1",
        "--output", str(output),
    ]
    env = {**os.environ,
           "MNEMOSYNE_DATA_DIR": str(ambient),
           "MNEMOSYNE_POLYPHONIC_RECALL": "1",
           "MNEMOSYNE_RECALL_CONTENT_CAP": "not-an-integer",
           "MNEMOSYNE_EMBEDDINGS_VIA_API": "1",
           "MNEMOSYNE_EMBEDDING_API_KEY": "must-not-appear-in-report",
           "PYTHONHASHSEED": "random"}
    result = subprocess.run(command, env=env, capture_output=True, text=True,
                            encoding="utf-8", timeout=60, check=True)
    report = json.loads(result.stdout)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert "must-not-appear-in-report" not in result.stdout
    assert "must-not-appear-in-report" not in result.stderr
    assert "must-not-appear-in-report" not in output.read_text(encoding="utf-8")
    assert list(ambient.iterdir()) == [config]
    assert config.read_text(encoding="utf-8") == "polyphonic_recall: true\nrecall_content_cap: 1\n"
    assert report["validation"]["engine"] == "linear"
    assert report["validation"]["seeded_rows"] == {"working_memory": 16, "episodic_memory": 16}
    assert report["settings"] == {**bench.SETTINGS, "PYTHONHASHSEED": "0"}
    assert report["workload"]["queries"] == list(bench.QUERIES)
    assert len(report["samples_ns"]) == 2
    assert all(len(batch) == 8 for batch in report["samples_ns"])
    assert report["summary"]["calls"] == 16
    assert all(0 < count <= 10 for batch in report["result_counts"] for count in batch)
    assert report["summary"] == bench.summarize(report["samples_ns"], report["round_elapsed_ns"])
    assert report["environment"]["machine_label"] == "pytest-machine"
    commit = report["revision"]["commit"]
    assert commit is None or (len(commit) == 40 and all(c in "0123456789abcdef" for c in commit))

    second = subprocess.run(command + ["--baseline", str(output)], env=env,
                            capture_output=True, text=True, encoding="utf-8",
                            timeout=60, check=True)
    compared = json.loads(second.stdout)
    assert compared["comparison"]["compatible"] is True
    assert "median_qps_change_pct" in compared["comparison"]

@pytest.fixture
def report():
    """Provide a minimal compatible reference report for comparison controls."""
    return {
        "schema_version": 1, "benchmark": "test", "harness_sha256": "hash",
        "environment": {}, "settings": {}, "validation": {}, "workload": {},
        "sqlite_pragmas": {}, "revision": {"commit": "reference", "dirty": False},
        "recorded_at_utc": "reference-time", "samples_ns": [[10]],
        "summary": {"median_qps": 100, "latency_ms": {"p50": 1, "p95": 2}},
    }


def test_compare_is_advisory_and_allows_revision_time_sample_changes(report):
    """Compatible revisions report slower, faster and unchanged latency deltas."""
    slower = copy.deepcopy(report)
    slower["summary"]["median_qps"] /= 2
    slower["summary"]["latency_ms"]["p50"] *= 1.5
    slower["summary"]["latency_ms"]["p95"] *= 2
    slower.update(revision={"commit": "candidate", "dirty": True},
                  recorded_at_utc="candidate-time", samples_ns=[[20]])
    assert bench.compare(slower, report)["compatible"] is True
    assert bench.compare(slower, report)["median_qps_change_pct"] == -50
    assert bench.compare(slower, report)["p50_latency_change_pct"] == 50
    assert bench.compare(slower, report)["p95_latency_change_pct"] == 100

    faster = copy.deepcopy(report)
    faster["summary"]["latency_ms"]["p50"] = 0.5
    faster["summary"]["latency_ms"]["p95"] = 1
    comparison = bench.compare(faster, report)
    assert comparison["p50_latency_change_pct"] == -50
    assert comparison["p95_latency_change_pct"] == -50

    identical = bench.compare(report, report)
    assert identical["p50_latency_change_pct"] == 0
    assert identical["p95_latency_change_pct"] == 0


@pytest.mark.parametrize("missing", [False, True])
def test_compare_rejects_changed_or_missing_metadata(report, missing):
    """A missing or changed compatibility field prevents performance comparison."""
    for field in ("workload", "environment", "settings", "validation", "sqlite_pragmas",
                  "benchmark", "harness_sha256"):
        changed = copy.deepcopy(report)
        if missing:
            del changed[field]
        else:
            changed[field] = None
        assert bench.compare(report, changed) == {
            "compatible": False, "mismatched_fields": [field],
        }


@pytest.mark.parametrize("error", [
    OSError("missing git"), subprocess.CalledProcessError(1, ["git"]),
    subprocess.TimeoutExpired(["git"], 10),
])
def test_revision_falls_back_without_git_or_on_timeout(monkeypatch, error):
    """Git failures yield explicit unknown metadata without aborting measurement."""
    def fail(*args, **kwargs):
        """Simulate a Git failure while checking its independent time limit."""
        assert kwargs["timeout"] == 10
        raise error
    monkeypatch.setattr(bench.subprocess, "run", fail)
    assert bench.revision() == {"commit": None, "dirty": None}


def test_direct_worker_requires_pinned_settings(monkeypatch):
    """An unprepared direct worker invocation cannot bypass isolation setup."""
    monkeypatch.delenv("MNEMOSYNE_POLYPHONIC_RECALL", raising=False)
    with pytest.raises(RuntimeError, match="isolate configuration"):
        bench.main(["--worker", "--machine", "test"])


@pytest.mark.parametrize("contents", [None, "not json", "[]", "{}",
                                      '{"revision": {}, "summary": null}'])
def test_baseline_errors_fail_before_worker(tmp_path, monkeypatch, capsys, contents):
    """Invalid baseline files fail before starting a worker or creating a report."""
    baseline = tmp_path / "baseline.json"
    if contents is not None:
        baseline.write_text(contents, encoding="utf-8")
    def unexpected(*args, **kwargs):
        """Fail the test if baseline validation accidentally launches a worker."""
        pytest.fail("invalid baseline must not start a worker")
    monkeypatch.setattr(bench.subprocess, "run", unexpected)
    output = tmp_path / "output.json"
    with pytest.raises(SystemExit) as exc:
        bench.main(["--machine", "test", "--baseline", str(baseline), "--output", str(output)])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "cannot load --baseline" in captured.err
    assert not captured.out
    assert not output.exists()


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "1"])
@pytest.mark.parametrize("metric", ["median_qps", "p50", "p95"])
def test_baseline_rejects_invalid_divisors(tmp_path, report, value, metric):
    """Comparison divisors must be finite positive numbers, excluding booleans."""
    summary = report["summary"] if metric == "median_qps" else report["summary"]["latency_ms"]
    summary[metric] = value
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="finite positive"):
        bench.load_baseline(path)


@pytest.mark.parametrize("timed_out", [False, True])
def test_worker_failure_publishes_no_report(tmp_path, monkeypatch, capsys, timed_out):
    """Worker errors and timeouts preserve stderr without publishing timing JSON."""
    def fail(command, **kwargs):
        """Simulate worker exit or timeout and verify the configured deadline."""
        assert kwargs["timeout"] == 7
        if timed_out:
            raise subprocess.TimeoutExpired(command, 7)
        return subprocess.CompletedProcess(command, 3, stdout="", stderr="worker failed\n")
    monkeypatch.setattr(bench.subprocess, "run", fail)
    output = tmp_path / "output.json"
    code = bench.main(["--machine", "test", "--timeout", "7", "--output", str(output)])
    assert code == (1 if timed_out else 3)
    captured = capsys.readouterr()
    assert ("timed out after 7 seconds" if timed_out else "worker failed") in captured.err
    assert not captured.out
    assert not output.exists()
