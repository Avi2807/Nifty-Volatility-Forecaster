"""Central paths, date boundaries, features, and training defaults."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
FORECASTS = ROOT / "forecasts"
REPORT = ROOT / "report"

BARS_PER_DAY = 75
MIN_BARS = 60
VAR_Z = 2.33
VAR_LEVEL = 0.01

DATA_START, DATA_END = "2015-01-01", "2023-06-30"
TRAIN_END = "2019-12-31"
VAL_START, VAL_END = "2020-01-01", "2020-12-31"
TEST_START, TEST_END = "2021-01-01", "2023-06-30"

BASE_FEATURES = [
    "log_rv", "log_rv_w", "log_rv_m", "log_parkinson",
    "ret", "neg_ret", "gap", "log_vix", "turnover_chg", "expiry_next",
    "dow_0", "dow_1", "dow_2", "dow_3",
]
SENT_FEATURES = ["sent_mean", "sent_neg_share", "sent_log_count"]
TARGET, TARGET_OC = "target_log_rv", "target_oc_ret"
HAR_X_EXTRA = ["log_vix", "neg_ret", "gap"]
WINDOW_SEQ, WINDOW_MLP = 22, 5
SEEDS = (0, 1, 2, 3, 4)
MODEL_NAMES = [
    "naive", "har", "har_x", "har_x_sent", "garch", "mlp", "lstm",
    "transformer_nosent", "transformer_sent",
]


@dataclass(frozen=True)
class TrainCfg:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 300
    patience: int = 15
    clip: float = 1.0
    dropout: float = 0.1
    warmup_epochs: int = 0


MLP_CFG = TrainCfg()
LSTM_CFG = TrainCfg()
TRF_CFG = TrainCfg(lr=5e-4, weight_decay=1e-3, warmup_epochs=5)
