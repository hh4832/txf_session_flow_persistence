from txf_session_flow_persistence.config import FULL_DATASET, NIGHT_DATASET, FIELDS, DEFAULT_INSTRUMENT
from txf_session_flow_persistence.source import fetch_finlab
import pytest


def test_six_field_requests_each_session(fake_vendor):
    calls = []
    def getter(key):
        calls.append(key)
        which, suffix = key.split(":", 1)
        session = "full_session" if which == FULL_DATASET else "afterhours"
        field = {v: k for k, v in FIELDS.items()}[suffix]
        return fake_vendor[session][field]
    for name in ("full_session", "afterhours"):
        raw = fetch_finlab(name, "2026-10-19", "2027-01-31", getter=getter)
        assert set(raw) == set(FIELDS)
        assert all(DEFAULT_INSTRUMENT in x.columns for x in raw.values())
    assert len(calls) == 12
    assert calls[0].startswith(FULL_DATASET)
    assert calls[6].startswith(NIGHT_DATASET)


def test_failure_does_not_fallback(fake_vendor):
    with pytest.raises(RuntimeError, match="retrieval failed"):
        fetch_finlab("afterhours", "2026-10-19", "2027-01-31", getter=lambda x: (_ for _ in ()).throw(RuntimeError("fake key")))
