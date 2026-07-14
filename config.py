import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-hilbert-plv-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "SMH", "SOXX",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "SMH", "SOXX",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

# Macro signals: (column, description, composite_weight, regime_sign)
# regime_sign: +1 = rising is risk-on, -1 = rising is risk-off
MACRO_SIGNALS = [
    ("VIX",       "CBOE Volatility Index",           0.30, -1),
    ("T10Y2Y",    "10Y-2Y Treasury Spread",          0.25, +1),
    ("DXY",       "US Dollar Index",                 0.20, -1),
    ("IG_SPREAD", "IG Credit Spread",                0.15, -1),
    ("HY_SPREAD", "HY Credit Spread",                0.10, -1),
]

# Rolling windows for PLV computation (trading days)
WINDOWS = [63, 252, 504, 1008, 2016, 4032, 4536]

# Bandpass filter applied before Hilbert transform
BANDPASS_ENABLED  = True
BANDPASS_LOW_CUT  = 0.02   # ~50-day lower period
BANDPASS_HIGH_CUT = 0.25   # ~4-day upper period

# Score mode: REGIME_WEIGHTED signs PLV by macro direction
SCORE_MODE = "REGIME_WEIGHTED"

MIN_SAMPLES = 32   # minimum window samples for valid PLV
TOP_N       = 3
