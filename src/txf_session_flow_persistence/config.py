"""Fixed v0.1.0 research contract. No portfolio/trading-signal logic."""
from dataclasses import dataclass
from datetime import date

FULL_DATASET = "futures_institutional_investors_trading_summary"
NIGHT_DATASET = "futures_institutional_investors_trading_summary_after_hours"
FIELDS = {
    "long": "多方交易口數",
    "short": "空方交易口數",
    "net": "多空交易口數淨額",
    "long_value": "多方交易契約金額(千元)",
    "short_value": "空方交易契約金額(千元)",
    "net_value": "多空交易契約金額淨額(千元)",
}
# Override this exact label if the user's authenticated FinLab column differs.
DEFAULT_INSTRUMENT = "臺股期貨_外資及陸資"
RAW_FOLDER_ID = "1913HOEClHF_GTuI54usuuTKmIlQOzU83"
RESULTS_FOLDER_ID = "1Plm1g2DORO2QBVt6lpxbef2sorne89AN"
START_DATE = date(2023, 10, 16)

@dataclass(frozen=True)
class ResearchConfig:
    start: str
    end: str
    instrument: str = DEFAULT_INSTRUMENT
    hac_lags: int = 5
    min_n: int = 30
    holdout_fraction: float = 0.30

    def validate(self):
        a, b = date.fromisoformat(self.start), date.fromisoformat(self.end)
        if a > b or a < START_DATE:
            raise ValueError("start/end invalid or earlier than FinLab night data history")
        if not self.instrument.strip() or not 0 < self.holdout_fraction < 1:
            raise ValueError("invalid instrument or holdout_fraction")
        if self.hac_lags < 0 or self.min_n < 1:
            raise ValueError("invalid analysis parameters")
        return self
