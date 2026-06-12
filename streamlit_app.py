import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
from datetime import date, timedelta
import config

st.set_page_config(page_title="Hilbert PLV Engine", layout="wide")

st.markdown("""
<style>
.main-header{font-size:2.3rem;font-weight:700;color:#1a1a2e;margin-bottom:0.2rem}
.sub-header{font-size:1rem;color:#555;margin-bottom:1.5rem}
.uni-title{font-size:1.3rem;font-weight:600;margin-top:1rem;margin-bottom:0.8rem;
           padding-left:0.5rem;border-left:5px solid #e94560}
.hero-card{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);
           color:white;border-radius:16px;padding:1.2rem;margin:0.4rem;text-align:center;
           box-shadow:0 6px 20px rgba(233,69,96,0.3)}
.win-card{background:linear-gradient(135deg,#0f3460 0%,#533483 100%);color:white;
          border-radius:16px;padding:1.2rem;margin:0.4rem;text-align:center;
          box-shadow:0 4px 12px rgba(83,52,131,0.3)}
.ticker{font-size:1.6rem;font-weight:800;letter-spacing:1px}
.score{font-size:0.9rem;margin-top:0.3rem;opacity:0.85}
.next-day{font-size:0.8rem;margin-top:0.2rem;opacity:0.7}
.plv-badge-high{background:#27ae60;border-radius:6px;padding:2px 8px;font-size:0.75rem;
                font-weight:700;color:white}
.plv-badge-med{background:#f39c12;border-radius:6px;padding:2px 8px;font-size:0.75rem;
               font-weight:700;color:white}
.plv-badge-low{background:#e74c3c;border-radius:6px;padding:2px 8px;font-size:0.75rem;
               font-weight:700;color:white}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">〜 Hilbert Phase Locking Value Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Instantaneous phase synchronisation between ETF returns and macro cycles · '
    'Hilbert transform · PLV(0→1) · Regime-weighted scoring · Multi-window</div>',
    unsafe_allow_html=True)

HF_TOKEN    = config.HF_TOKEN
OUTPUT_REPO = config.OUTPUT_REPO

US_HOLIDAYS = {
    date(2025,1,1),date(2025,1,20),date(2025,2,17),date(2025,4,18),
    date(2025,5,26),date(2025,6,19),date(2025,7,4),date(2025,9,1),
    date(2025,11,27),date(2025,12,25),
    date(2026,1,1),date(2026,1,19),date(2026,2,16),date(2026,4,3),
    date(2026,5,25),date(2026,6,19),date(2026,7,3),date(2026,9,7),
    date(2026,11,26),date(2026,12,25),
}

def next_trading_day() -> str:
    d = date.today() + timedelta(days=1)
    while d.weekday() >= 5 or d in US_HOLIDAYS:
        d += timedelta(days=1)
    return d.strftime("%B %d, %Y")

def plv_badge(score: float) -> str:
    if score > 0.5:   return f'<span class="plv-badge-high">HIGH SYNC</span>'
    elif score > 0.0: return f'<span class="plv-badge-med">MED SYNC</span>'
    else:             return f'<span class="plv-badge-low">LOW SYNC</span>'


@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        return [f["name"] for f in fs.ls(f"datasets/{OUTPUT_REPO}",
                                          detail=True, recursive=True)
                if f["type"] == "file"]
    except Exception as e:
        return []


def find_latest(files, prefix):
    matches = sorted([f for f in files if f.endswith(".json") and prefix in f], reverse=True)
    return matches[0] if matches else None


@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 〜 Hilbert PLV")
st.sidebar.markdown(f"**Next Trading Day**")
st.sidebar.markdown(f"`{next_trading_day()}`")
st.sidebar.markdown(f"**Windows:** {config.WINDOWS}")
st.sidebar.markdown(f"**Mode:** {config.SCORE_MODE}")
st.sidebar.markdown(f"**Bandpass:** {'✅' if config.BANDPASS_ENABLED else '❌'} "
                    f"({config.BANDPASS_LOW_CUT}–{config.BANDPASS_HIGH_CUT} cyc/day)")
st.sidebar.markdown("**Macro signals:**")
for col, desc, w, sign in config.MACRO_SIGNALS:
    arrow = "↑risk-on" if sign > 0 else "↑risk-off"
    st.sidebar.markdown(f"  • {col} ({arrow}, w={w:.0%})")

# ── Load data ─────────────────────────────────────────────────────────────────
files     = list_repo_files()
tab1_path = find_latest(files, "hilbert_plv_2")
tab2_path = find_latest(files, "hilbert_plv_windows_")

if not tab1_path:
    st.error("No results found. Run trainer.py first.")
    st.stop()

data1 = load_json(tab1_path)
if "error" in data1:
    st.error(f"Error loading data: {data1['error']}")
    st.stop()

data2      = load_json(tab2_path) if tab2_path else None
universes1 = data1["universes"]
universes2 = data2["universes"] if data2 and "error" not in data2 else None

st.sidebar.markdown(f"**Run date:** `{data1.get('run_date','?')}`")

tab1, tab2 = st.tabs(["🏆 Best Window per ETF", "🔍 Explore by Window"])

UNIVERSE_ORDER  = ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]
UNIVERSE_LABELS = {
    "FI_COMMODITIES": "🏦 FI & Commodities",
    "EQUITY_SECTORS": "📈 Equity Sectors",
    "COMBINED":       "🌐 Combined",
}

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 Top ETFs — Phase-Synchronised with Macro Cycles")

    with st.expander("📖 How Hilbert PLV Works", expanded=True):
        st.markdown("""
**Phase Locking Value (PLV)** measures how synchronised the rhythm of an ETF's returns is
with a macro indicator's rhythm.

| Step | What happens |
|------|-------------|
| 1. Bandpass filter | Removes trend and noise; isolates the cyclical component |
| 2. Hilbert transform | Extracts the *instantaneous phase* φ(t) of each signal |
| 3. PLV = \|mean(exp(j·Δφ))\| | Measures phase coherence: 1 = locked, 0 = random |
| 4. Regime weighting | PLV is signed by macro direction (rising VIX = bad → negative score) |
| 5. Composite | Weighted sum across VIX, T10Y2Y, DXY, IG_SPREAD, HY_SPREAD |

**HIGH PLV + risk-on macro** → ETF is rhythmically synced to a favourable macro cycle → **BUY signal**

**HIGH PLV + risk-off macro** → ETF synced to danger → **AVOID**
        """)

    ntd = next_trading_day()

    for universe_name in UNIVERSE_ORDER:
        uni_data = universes1.get(universe_name, {})
        top_etfs = uni_data.get("top_etfs", [])
        if not top_etfs:
            continue

        label = UNIVERSE_LABELS.get(universe_name, universe_name)
        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        cols = st.columns(3)
        for idx, etf in enumerate(top_etfs):
            score  = etf["plv_score"]
            win    = etf.get("best_window", "N/A")
            badge  = plv_badge(score)
            with cols[idx]:
                st.markdown(f"""
<div class="hero-card">
  <div class="ticker">{etf['ticker']}</div>
  <div class="score">PLV score = {score:+.4f}</div>
  <div class="score">{badge}</div>
  <div class="score">best window = {win}d</div>
  <div class="next-day">📅 {ntd}</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"📋 Full ranking — {label}"):
            full = uni_data.get("full_scores", {})
            if full:
                rows = [{"ETF": t,
                         "PLV Score": round(info.get("score", info) if isinstance(info, dict) else info, 4),
                         "Best Window (d)": info.get("best_window","N/A") if isinstance(info, dict) else "N/A"}
                        for t, info in full.items()]
                df_rank = pd.DataFrame(rows).sort_values("PLV Score", ascending=False)
                st.dataframe(df_rank, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(f"Run date: {data1.get('run_date','?')} · "
               "Hilbert transform + Phase Locking Value · "
               "Regime-weighted composite score · Cross-sectional z-score")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Explore PLV Rankings by Window")

    if not universes2:
        st.warning("Window-level data not found. Re-run trainer.")
        st.stop()

    all_wins = set()
    for ud in universes2.values():
        all_wins.update(ud.get("windows", {}).keys())
    win_options = sorted([int(w) for w in all_wins])

    if not win_options:
        st.error("No window data.")
        st.stop()

    win_labels = {
        63:   "63d  (~3 months)",
        252:  "252d (~1 year)",
        504:  "504d (~2 years)",
        1008: "1008d (~4 years)",
        2016: "2016d (~8 years)",
        4032: "4032d (~16 years)",
        4536: "4536d (~18 years)",
    }

    default_idx  = win_options.index(252) if 252 in win_options else 0
    selected_win = st.selectbox(
        "Select lookback window",
        options=win_options,
        index=default_idx,
        format_func=lambda w: win_labels.get(w, f"{w}d"),
    )
    win_key = str(selected_win)

    with st.expander("ℹ️ Window guidance", expanded=False):
        st.markdown("""
- **63d** — Short-term phase sync: captures recent macro rhythm coupling
- **252d** — Annual cycle alignment: recommended primary signal
- **504d–1008d** — Medium-term structural sync regimes
- **2016d+** — Very long-run macro-ETF phase relationships (secular cycles)
- **4032d / 4536d** — Full history phase coherence (2008–present)
        """)

    st.markdown(f"### PLV Rankings at **{win_labels.get(selected_win, f'{selected_win}d')}** window")

    for universe_name in UNIVERSE_ORDER:
        label    = UNIVERSE_LABELS.get(universe_name, universe_name)
        uni_data = universes2.get(universe_name, {})
        win_data = uni_data.get("windows", {}).get(win_key)

        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        if not win_data:
            st.info(f"No data for {universe_name} at {selected_win}d.")
            st.divider()
            continue

        cols = st.columns(3)
        for idx, etf in enumerate(win_data.get("top_etfs", [])):
            score = etf["plv_score"]
            badge = plv_badge(score)
            with cols[idx]:
                st.markdown(f"""
<div class="win-card">
  <div class="ticker">{etf['ticker']}</div>
  <div class="score">PLV score = {score:+.4f}</div>
  <div class="score">{badge}</div>
  <div class="next-day">window = {selected_win}d · 📅 {ntd}</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"📋 Full ranking — {label} @ {selected_win}d"):
            rows = win_data.get("full_ranking", [])
            if rows:
                df_win = pd.DataFrame(rows)
                df_win.columns = ["ETF", "PLV Score"]
                df_win.insert(0, "Rank", range(1, len(df_win) + 1))
                st.dataframe(df_win, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(f"Window: {selected_win}d · Run date: {data2.get('run_date','?')}")
