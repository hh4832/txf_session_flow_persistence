import json
import pytest
from txf_session_flow_persistence.storage import LocalStore, snapshot_path, checksum


def test_cache_never_overwrites(tmp_path):
    store = LocalStore(tmp_path / "raw", tmp_path / "results")
    store.write_raw("full_session/a.txt", b"original")
    store.write_raw("full_session/a.txt", b"original")
    assert store.read_raw("full_session/a.txt") == b"original"
    with pytest.raises(FileExistsError):
        store.write_raw("full_session/a.txt", b"replaced")
    assert store.read_raw("full_session/a.txt") == b"original"


def test_immutable_result_runs(tmp_path):
    store = LocalStore(tmp_path / "raw", tmp_path / "results")
    store.write_result("runs/1/tables/out.csv", b"v1")
    store.write_result("runs/2/tables/out.csv", b"v2")
    with pytest.raises(FileExistsError):
        store.write_result("runs/1/tables/out.csv", b"v2")


def test_distinct_drive_contract_paths():
    assert snapshot_path("full_session", "2023-10-16", "2026-10-22").startswith("full_session/")
    assert snapshot_path("afterhours", "2023-10-16", "2026-10-22", "new") != snapshot_path("afterhours", "2023-10-16", "2026-10-22")
    with pytest.raises(ValueError):
        snapshot_path("oi", "2023-10-16", "2026-10-22")
