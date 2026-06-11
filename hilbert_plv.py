"""
Phase Synchronization with Macro — Hilbert Phase Locking Value (PLV)
======================================================================
Uses the Hilbert transform to extract the instantaneous phase of both
ETF returns and macro signals, then computes the Phase Locking Value
(PLV) between them over a rolling window.

Theory
------
Given a real-valued signal x(t), the analytic signal is:
    z(t) = x(t) + j·H{x(t)}
where H{·} is the Hilbert transform. The instantaneous phase is:
    φ(t) = arctan2(Im(z(t)), Re(z(t)))

The Phase Locking Value between two signals over N samples is:
    PLV = |1/N · Σ exp(j·(φ_etf(t) − φ_macro(t)))|

PLV ∈ [0, 1]:
    1 = perfect phase synchronisation (ETF and macro cycle in lockstep)
    0 = no phase relationship (random phase differences)

Scoring
-------
Raw PLV is direction-neutral. We weight it by the macro regime sign:
    signed_PLV = PLV × sign(Δmacro_recent) × regime_sign

This means:
  - High PLV with rising VIX (risk-off macro) → NEGATIVE score
    (ETF synced to a bad macro cycle = avoid)
  - High PLV with steepening yield curve (risk-on) → POSITIVE score
    (ETF synced to a good macro cycle = overweight)

Composite score = weighted sum across all macro signals, cross-sectional z-score.

References
----------
Lachaux, J.P. et al. (1999). Measuring phase synchrony in brain signals.
    Human Brain Mapping, 8(4), 194-208.
Pikovsky, A., Rosenblum, M., Kurths, J. (2001). Synchronization. Cambridge UP.
"""

import numpy as np
import pandas as pd
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import zscore as sp_zscore


# ── Signal pre-processing ─────────────────────────────────────────────────────

def _bandpass_filter(series: np.ndarray, low_cut: float, high_cut: float,
                     fs: float = 1.0, order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to remove trend (low-freq) and
    microstructure noise (high-freq) before phase extraction.

    Parameters
    ----------
    series   : 1-D float array
    low_cut  : lower cutoff frequency (cycles per sample, e.g. 0.02 ≈ 50-day)
    high_cut : upper cutoff frequency (cycles per sample, e.g. 0.25 ≈ 4-day)
    fs       : sampling frequency (1.0 for daily data)
    order    : Butterworth filter order

    Returns
    -------
    Filtered 1-D float array, same length as input.
    """
    nyq = 0.5 * fs
    low  = low_cut  / nyq
    high = high_cut / nyq
    low  = max(1e-4, min(low,  0.999))
    high = max(1e-4, min(high, 0.999))
    if low >= high:
        return series
    try:
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, series)
    except Exception:
        return series


def _extract_phase(series: np.ndarray,
                   bandpass: bool = True,
                   low_cut: float = 0.02,
                   high_cut: float = 0.25) -> np.ndarray:
    """
    Extract instantaneous phase via Hilbert transform.

    Steps:
      1. Fill NaN with 0 (mean-imputed — phase only needs relative values)
      2. Optionally bandpass filter
      3. Compute analytic signal via Hilbert transform
      4. Extract phase with arctan2

    Returns phase array in radians, same length as input.
    """
    x = np.where(np.isfinite(series), series, 0.0)
    if bandpass and len(x) > 16:
        x = _bandpass_filter(x, low_cut, high_cut)
    analytic = hilbert(x)
    return np.angle(analytic)   # arctan2(imag, real) ∈ [-π, π]


# ── Phase Locking Value ───────────────────────────────────────────────────────

def compute_plv(phase_a: np.ndarray, phase_b: np.ndarray) -> float:
    """
    Compute Phase Locking Value between two phase sequences.

        PLV = |mean(exp(j·(φ_a − φ_b)))|

    Parameters
    ----------
    phase_a, phase_b : 1-D arrays of instantaneous phase (radians)

    Returns
    -------
    PLV ∈ [0, 1]
    """
    if len(phase_a) != len(phase_b) or len(phase_a) < 2:
        return 0.0
    delta_phase = phase_a - phase_b
    plv = np.abs(np.mean(np.exp(1j * delta_phase)))
    return float(np.clip(plv, 0.0, 1.0))


def rolling_plv(returns_etf: pd.Series,
                macro_series: pd.Series,
                window: int,
                bandpass: bool = True,
                low_cut: float = 0.02,
                high_cut: float = 0.25,
                min_samples: int = 32) -> pd.Series:
    """
    Compute rolling PLV between an ETF return series and a macro series.

    Parameters
    ----------
    returns_etf  : daily log returns for one ETF
    macro_series : macro signal (level or change — both work)
    window       : rolling window in days
    bandpass     : apply bandpass filter before phase extraction
    low_cut/high_cut : bandpass cutoffs
    min_samples  : minimum non-NaN samples required in window

    Returns
    -------
    pd.Series of PLV values (0-1), indexed by date.
    NaN where insufficient data.
    """
    common = returns_etf.index.intersection(macro_series.index)
    if len(common) < min_samples:
        return pd.Series(np.nan, index=returns_etf.index)

    r_etf  = returns_etf.reindex(common).fillna(0.0).values.astype(float)
    r_mac  = macro_series.reindex(common).fillna(0.0).values.astype(float)
    dates  = common
    n      = len(r_etf)
    plv_vals = np.full(n, np.nan)

    for i in range(window - 1, n):
        start = max(0, i - window + 1)
        seg_e = r_etf[start: i + 1]
        seg_m = r_mac[start: i + 1]
        valid = np.isfinite(seg_e) & np.isfinite(seg_m)
        if valid.sum() < min_samples:
            continue
        ph_e = _extract_phase(seg_e, bandpass, low_cut, high_cut)
        ph_m = _extract_phase(seg_m, bandpass, low_cut, high_cut)
        plv_vals[i] = compute_plv(ph_e, ph_m)

    return pd.Series(plv_vals, index=dates)


# ── Composite score ───────────────────────────────────────────────────────────

def compute_plv_scores(prices: pd.DataFrame,
                       macro_df: pd.DataFrame,
                       tickers: list,
                       macro_signals: list,
                       window: int,
                       score_mode: str = "REGIME_WEIGHTED",
                       bandpass: bool = True,
                       low_cut: float = 0.02,
                       high_cut: float = 0.25,
                       min_samples: int = 32) -> pd.Series:
    """
    Compute composite PLV score for each ETF in tickers.

    Parameters
    ----------
    prices       : DataFrame of ETF closing prices
    macro_df     : DataFrame containing macro columns
    tickers      : list of ETF tickers to score
    macro_signals: list of (col, desc, weight, regime_sign) tuples from config
    window       : rolling window in days
    score_mode   : "REGIME_WEIGHTED" or "PURE_PLV"
    bandpass     : apply bandpass before Hilbert
    low_cut/high_cut : bandpass cutoffs
    min_samples  : minimum samples for valid PLV

    Returns
    -------
    pd.Series {ticker: composite_score}, cross-sectionally z-scored.
    """
    avail = [t for t in tickers if t in prices.columns]
    if not avail:
        return pd.Series(dtype=float)

    # Compute log returns for ETFs
    log_ret = np.log(prices[avail] / prices[avail].shift(1))

    # Compute macro changes (daily pct change of level — stationarises VIX, DXY etc)
    macro_changes = {}
    for col, _, weight, sign in macro_signals:
        if col in macro_df.columns:
            macro_changes[col] = macro_df[col].ffill().pct_change(fill_method=None)

    if not macro_changes:
        return pd.Series(dtype=float)

    # For regime direction: use recent 21-day change sign of each macro
    recent_macro_direction = {}
    for col, _, weight, sign in macro_signals:
        if col in macro_df.columns:
            recent = macro_df[col].ffill().iloc[-21:]
            if len(recent) >= 2:
                direction = np.sign(recent.iloc[-1] - recent.iloc[0])
                recent_macro_direction[col] = float(direction)
            else:
                recent_macro_direction[col] = 0.0

    composite = {}

    for ticker in avail:
        etf_ret  = log_ret[ticker].dropna()
        if len(etf_ret) < min_samples:
            continue

        total_score  = 0.0
        total_weight = 0.0

        for col, _, weight, regime_sign in macro_signals:
            if col not in macro_changes:
                continue
            mac_series = macro_changes[col].dropna()
            if len(mac_series) < min_samples:
                continue

            plv_series = rolling_plv(
                etf_ret, mac_series, window,
                bandpass, low_cut, high_cut, min_samples
            )
            plv_latest = plv_series.dropna()
            if plv_latest.empty:
                continue
            plv_val = float(plv_latest.iloc[-1])

            if score_mode == "REGIME_WEIGHTED":
                # Sign the PLV by: macro regime direction × regime sign
                macro_dir = recent_macro_direction.get(col, 0.0)
                signed    = plv_val * macro_dir * regime_sign
            else:
                signed = plv_val

            total_score  += weight * signed
            total_weight += weight

        if total_weight > 0:
            composite[ticker] = total_score / total_weight

    if not composite:
        return pd.Series(dtype=float)

    scores = pd.Series(composite).dropna()
    if len(scores) < 2:
        return scores

    # Cross-sectional z-score
    mu, sd = scores.mean(), scores.std()
    if sd < 1e-10:
        return pd.Series(0.0, index=scores.index)
    return (scores - mu) / sd
