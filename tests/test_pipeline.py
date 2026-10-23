"""End-to-end offline smoke under real pyarrow when dependency exists in CI."""
from pathlib import Path
import pytest
from txf_session_flow_persistence.config import FIELDS, FULL_DATASET, NIGHT_DATASET, ResearchConfig
from txf_session_flow_persistence.runner import run
from txf_session_flow_persistence.storage import LocalStore


def test_offline_pipeline_cache_and_immutable_runs(fake_vendor, tmp_path):
    pytest.importorskip("pyarrow")
    store = LocalStore(tmp_path / "raw", tmp_path / "results")
    requests = []
    inverse = {v:k for k,v in FIELDS.items()}
    def getter(key):
        requests.append(key)
        dataset, field = key.split(":", 1)
        return fake_vendor["full_session" if dataset == FULL_DATASET else "afterhours"][inverse[field]]
    config = ResearchConfig(start="2026-10-19", end="2027-01-31", min_n=5)
    first, tables = run(store, config, getter=getter, compute_figure=False)
    assert first["status"] == "ANALYSIS_COMPLETE" and first["data_type"] == "INJECTED_TEST_DATA"
    assert len(requests) == 12
    assert (tmp_path / "raw/full_session/2026-10-19_2027-01-31/manifest.json").exists()
    assert (tmp_path / "results" / first["run_name"] / "reports/summary.md").exists()
    def must_not_fetch(_):
        raise AssertionError("cache required")
    second, _ = run(store, config, getter=must_not_fetch, compute_figure=False)
    assert len(requests) == 12
    assert first["run_name"] != second["run_name"]
    assert (tmp_path / "results" / first["run_name"] / "run_info.json").exists()
    assert (tmp_path / "results" / second["run_name"] / "run_info.json").exists()
