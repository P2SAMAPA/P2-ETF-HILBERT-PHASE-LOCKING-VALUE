"""
trainer.py  —  Hilbert PLV Engine orchestrator
===============================================

1. Load master dataset via data_manager.py
2. For every universe × window: compute composite PLV scores via hilbert_plv.py
3. Build two JSON result files:
     hilbert_plv_YYYY-MM-DD.json          → Tab 1  (best window per ETF)
     hilbert_plv_windows_YYYY-MM-DD.json  → Tab 2  (all windows per universe)
4. Upload both files to HuggingFace via push_results.py

JSON schema — Tab 1  (hilbert_plv_YYYY-MM-DD.json)
----------------------------------------------------
{
  "run_date": "YYYY-MM-DD",
  "universes": {
    "FI_COMMODITIES": {
      "top_etfs": [
        {"ticker": "TLT", "plv_score": 0.42, "best_window": 252},
        ...
      ],
      "full_scores": {
        "TLT": {"score": 0.42, "best_window": 252},
        ...
      }
    },
    ...
  }
}

JSON schema — Tab 2  (hilbert_plv_windows_YYYY-MM-DD.json)
-----------------------------------------------------------
{
  "run_date": "YYYY-MM-DD",
  "universes": {
    "FI_COMMODITIES": {
      "windows": {
        "63":  {"top_etfs": [...], "full_ranking": [[ticker, score], ...]},
        "252": {"top_etfs": [...], "full_ranking": [[ticker, score], ...]},
        ...
      }
    },
    ...
  }
}
"""

import json
import logging
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import config
import data_manager
import push_results
from hilbert_plv import compute_plv_scores

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(val) -> float:
    """JSON-safe float: map NaN / Inf → 0.0."""
    try:
        f = float(val)
        return f if np.isfinite(f) else 0.0
    except Exception:
        return 0.0


def _build_top_etfs(scores: pd.Series, top_n: int,
                    best_window: dict = None) -> list:
    """
    Return the top_n ETFs sorted by score descending.
    best_window: optional {ticker: window_int} for Tab 1 cards.
    """
    ranked = scores.sort_values(ascending=False)
    result = []
    for ticker in ranked.index[:top_n]:
        entry = {
            "ticker":    ticker,
            "plv_score": _safe_float(ranked[ticker]),
        }
        if best_window is not None:
            entry["best_window"] = best_window.get(ticker, "N/A")
        result.append(entry)
    return result


def _build_full_ranking(scores: pd.Series) -> list:
    """[[ticker, score], ...] sorted descending — for Tab 2 full_ranking."""
    ranked = scores.sort_values(ascending=False)
    return [[t, _safe_float(s)] for t, s in ranked.items()]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    run_date = date.today().isoformat()
    logger.info(f"=== Hilbert PLV Engine  |  {run_date} ===")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    token = config.HF_TOKEN or os.environ.get("HF_TOKEN", "")
    prices, macro = data_manager.load_master_data(hf_token=token)
    data_manager.validate_data(prices, macro)

    # ── 2. Compute PLV per universe × window ──────────────────────────────────
    # all_scores[universe][window] = pd.Series({ticker: composite_z_score})
    all_scores: dict[str, dict[int, pd.Series]] = {}

    for universe_name, tickers in config.UNIVERSES.items():
        logger.info(f"Universe: {universe_name}  ({len(tickers)} tickers)")
        all_scores[universe_name] = {}

        for window in config.WINDOWS:
            logger.info(f"  window={window}d …")
            try:
                scores = compute_plv_scores(
                    prices        = prices,
                    macro_df      = macro,
                    tickers       = tickers,
                    macro_signals = config.MACRO_SIGNALS,
                    window        = window,
                    score_mode    = config.SCORE_MODE,
                    bandpass      = config.BANDPASS_ENABLED,
                    low_cut       = config.BANDPASS_LOW_CUT,
                    high_cut      = config.BANDPASS_HIGH_CUT,
                    min_samples   = config.MIN_SAMPLES,
                )
            except Exception as exc:
                logger.warning(f"  ⚠️  {universe_name} w={window}: {exc}")
                scores = pd.Series(dtype=float)

            all_scores[universe_name][window] = scores

            if not scores.empty:
                top3 = scores.nlargest(3).index.tolist()
                logger.info(f"  → top 3: {top3}")
            else:
                logger.warning(f"  → empty scores for {universe_name} w={window}")

    # ── 3. Build Tab 1 JSON — best window per ETF ─────────────────────────────
    tab1_universes = {}

    for universe_name, tickers in config.UNIVERSES.items():
        best_window_map: dict[str, int]   = {}  # ticker → window with highest |score|
        best_score_map:  dict[str, float] = {}  # ticker → best score value

        for ticker in tickers:
            best_abs  = -1.0
            best_win  = None
            best_val  = 0.0
            for window, scores in all_scores[universe_name].items():
                if ticker in scores.index:
                    val = _safe_float(scores[ticker])
                    if abs(val) > best_abs:
                        best_abs = abs(val)
                        best_win = window
                        best_val = val
            if best_win is not None:
                best_window_map[ticker] = best_win
                best_score_map[ticker]  = best_val

        if not best_score_map:
            logger.warning(f"No scores for {universe_name} — empty Tab 1 entry.")
            tab1_universes[universe_name] = {"top_etfs": [], "full_scores": {}}
            continue

        best_series = pd.Series(best_score_map)
        top_etfs    = _build_top_etfs(best_series, config.TOP_N, best_window_map)
        full_scores = {
            t: {
                "score":       _safe_float(s),
                "best_window": best_window_map.get(t, "N/A"),
            }
            for t, s in best_series.sort_values(ascending=False).items()
        }

        tab1_universes[universe_name] = {
            "top_etfs":    top_etfs,
            "full_scores": full_scores,
        }

    tab1_payload = {
        "run_date":  run_date,
        "universes": tab1_universes,
    }

    # ── 4. Build Tab 2 JSON — all windows per universe ────────────────────────
    tab2_universes = {}

    for universe_name in config.UNIVERSES:
        windows_dict = {}
        for window, scores in all_scores[universe_name].items():
            if scores.empty:
                windows_dict[str(window)] = {"top_etfs": [], "full_ranking": []}
            else:
                windows_dict[str(window)] = {
                    "top_etfs":    _build_top_etfs(scores, config.TOP_N),
                    "full_ranking": _build_full_ranking(scores),
                }
        tab2_universes[universe_name] = {"windows": windows_dict}

    tab2_payload = {
        "run_date":  run_date,
        "universes": tab2_universes,
    }

    # ── 5. Write JSON files ───────────────────────────────────────────────────
    tab1_path = Path(f"hilbert_plv_{run_date}.json")
    tab2_path = Path(f"hilbert_plv_windows_{run_date}.json")

    with open(tab1_path, "w") as f:
        json.dump(tab1_payload, f, indent=2)
    logger.info(f"Wrote {tab1_path}")

    with open(tab2_path, "w") as f:
        json.dump(tab2_payload, f, indent=2)
    logger.info(f"Wrote {tab2_path}")

    # ── 6. Upload to HuggingFace ──────────────────────────────────────────────
    push_results.push_daily_result(tab1_path)
    push_results.push_daily_result(tab2_path)

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
