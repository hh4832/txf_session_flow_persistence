import pytest
import numpy as np
import pandas as pd
from txf_session_flow_persistence.config import FIELDS, FULL_DATASET, NIGHT_DATASET, DEFAULT_INSTRUMENT


@pytest.fixture
def fake_vendor():
    """Synthetic and clearly labeled; never claimed to be financial observations."""
    days = pd.bdate_range("2026-10-19", periods=60)
    rng = np.random.default_rng(42)
    # each day flow changes; all counts nonnegative, full = day + night.
    nl = rng.integers(1000, 3000, len(days))
    ns = rng.integers(1000, 3000, len(days))
    dl = rng.integers(3000, 5000, len(days))
    ds = rng.integers(3000, 5000, len(days))
    vals = {
        "afterhours": {"long": nl, "short": ns, "net": nl-ns, "long_value": nl*100, "short_value": ns*100, "net_value": (nl-ns)*100},
        "full_session": {"long": nl+dl, "short": ns+ds, "net": nl+dl-ns-ds,
                         "long_value": (nl+dl)*100, "short_value": (ns+ds)*100, "net_value": (nl+dl-ns-ds)*100}
    }
    raw = {}
    for session, fields in vals.items():
        raw[session] = {}
        for field, data in fields.items():
            raw[session][field] = pd.DataFrame({DEFAULT_INSTRUMENT: data,
                                               "小型臺指期貨_外資及陸資": data * 3}, index=days)
    return raw
