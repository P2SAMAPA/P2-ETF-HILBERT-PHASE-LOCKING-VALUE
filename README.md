# 〜 P2-ETF-HILBERT-PHASE-LOCKING-VALUE

**Phase Synchronisation Engine — Hilbert Transform + Phase Locking Value (PLV)**

Part of the **P2Quant Engine Suite** · [P2SAMAPA](https://github.com/P2SAMAPA)

---

## What This Engine Does

This engine measures how *rhythmically synchronised* ETF return cycles are with macro indicator cycles, using signal-processing techniques borrowed from neuroscience.

### Theory

Given a real-valued signal x(t), the **analytic signal** is:

```
z(t) = x(t) + j·H{x(t)}
```

where H{·} is the Hilbert transform. The **instantaneous phase** is:

```
φ(t) = arctan2(Im(z(t)), Re(z(t)))
```

The **Phase Locking Value** between an ETF and a macro signal over N samples is:

```
PLV = |1/N · Σ exp(j·(φ_etf(t) − φ_macro(t)))|
```

PLV ∈ [0, 1]:
- **1** = perfect phase synchronisation (ETF and macro cycle in lockstep)
- **0** = no phase relationship (random phase differences)

### Regime Weighting

Raw PLV is direction-neutral. We sign it by macro regime direction:

```
signed_PLV = PLV × sign(Δmacro_recent) × regime_sign
```

| Scenario | Interpretation | Score direction |
|---|---|---|
| High PLV + rising VIX (risk-off) | ETF synced to bad macro cycle | Negative → avoid |
| High PLV + steepening curve (risk-on) | ETF synced to good macro cycle | Positive → overweight |
| Low PLV | No macro phase coupling | Near zero |

Composite score = weighted sum across all macro signals → cross-sectional z-score.

---

## Universes

| Universe | Tickers |
|---|---|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

---

## Macro Signals

| Column | Description | Weight | Regime |
|---|---|---|---|
| VIX | CBOE Volatility Index | 30% | ↑ = risk-off |
| T10Y2Y | 10Y–2Y Treasury Spread | 25% | ↑ = risk-on |
| DXY | US Dollar Index | 20% | ↑ = risk-off |
| IG_SPREAD | IG Credit Spread | 15% | ↑ = risk-off |
| HY_SPREAD | HY Credit Spread | 10% | ↑ = risk-off |

---

## Rolling Windows

```
63d · 252d · 504d · 1008d · 2016d · 4032d · 4536d
```

(~3 months to ~18 years of trading days)

---

## Repository Structure

```
P2-ETF-HILBERT-PHASE-LOCKING-VALUE/
├── config.py           # Universes, macro signals, windows, hyperparams
├── data_manager.py     # HuggingFace data loader → (prices, macro) DataFrames
├── hilbert_plv.py      # Core signal processing: bandpass → Hilbert → PLV → scores
├── trainer.py          # Orchestrator: load → score → build JSON → upload to HF
├── push_results.py     # HfApi.upload_file wrapper for HuggingFace output repo
├── streamlit_app.py    # Two-tab Streamlit dashboard
├── us_calendar.py      # US trading calendar helper
├── requirements.txt    # Python dependencies
└── .github/
    └── workflows/
        └── daily.yml   # Scheduled GitHub Actions run (00:30 UTC Mon–Sat)
```

---

## Data Flow

```
HuggingFace Master Dataset
P2SAMAPA/fi-etf-macro-signal-master-data
           │
           ▼
     data_manager.py
    (prices, macro DFs)
           │
           ▼
      hilbert_plv.py
  bandpass → Hilbert phase →
  rolling PLV → regime sign →
  composite z-score
           │
           ▼
       trainer.py
  builds two JSON files:
  • hilbert_plv_YYYY-MM-DD.json
  • hilbert_plv_windows_YYYY-MM-DD.json
           │
           ▼
     push_results.py
  HfApi.upload_file →
P2SAMAPA/p2-etf-hilbert-plv-results
           │
           ▼
    streamlit_app.py
  Tab 1: Best Window per ETF
  Tab 2: Explore by Window
```

---

## Output JSON Schemas

### Tab 1 — `hilbert_plv_YYYY-MM-DD.json`

```json
{
  "run_date": "2026-06-18",
  "universes": {
    "FI_COMMODITIES": {
      "top_etfs": [
        {"ticker": "TLT", "plv_score": 0.42, "best_window": 252}
      ],
      "full_scores": {
        "TLT": {"score": 0.42, "best_window": 252}
      }
    }
  }
}
```

### Tab 2 — `hilbert_plv_windows_YYYY-MM-DD.json`

```json
{
  "run_date": "2026-06-18",
  "universes": {
    "FI_COMMODITIES": {
      "windows": {
        "63":  {"top_etfs": [...], "full_ranking": [["TLT", 0.42], ...]},
        "252": {"top_etfs": [...], "full_ranking": [...]}
      }
    }
  }
}
```

---

## Setup & Local Run

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-HILBERT-PHASE-LOCKING-VALUE
cd P2-ETF-HILBERT-PHASE-LOCKING-VALUE
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py
```

### Streamlit dashboard

```bash
streamlit run streamlit_app.py
```

---

## GitHub Actions

Runs automatically at **00:30 UTC Monday–Saturday** via `.github/workflows/daily.yml`.

Required secret: `HF_TOKEN` (set in repo Settings → Secrets → Actions).

`workflow_dispatch` is enabled for manual runs from the Actions tab.

---

## Key Implementation Notes

- **No `dropna()` on all columns** — only `dropna(subset=MACRO_COLS_CORE)` to avoid losing history from extended macro columns with variable start dates.
- **`HfApi.upload_file`** used for all HuggingFace writes (not `HfFileSystem.open`).
- **`HF_TOKEN`** passed on every step touching HuggingFace, not just the upload step.
- **Log returns** computed as `log(price_t / price_{t-1})` from raw price columns — the master parquet contains prices, not pre-computed returns.
- **Cross-sectional z-score** applied per universe per window so scores are comparable across different ETF sets.

---

## References

- Lachaux, J.P. et al. (1999). *Measuring phase synchrony in brain signals.* Human Brain Mapping, 8(4), 194–208.
- Pikovsky, A., Rosenblum, M., Kurths, J. (2001). *Synchronization: A Universal Concept in Nonlinear Sciences.* Cambridge University Press.
- Gabor, D. (1946). *Theory of communication.* Journal of IEE, 93(3), 429–457.
