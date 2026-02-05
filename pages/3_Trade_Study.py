from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
BASELINE = OUTPUTS / "baseline"
SWEEP = OUTPUTS / "sweep"
PLOTS_COMPARE = ROOT / "plots_compare_arrangements"

st.title("Trade Study / Sweeps")
st.caption("Precomputed artifacts (fast, safe for Streamlit Cloud).")

st.subheader("Baseline run")
baseline_imgs = [
    "accuracy_vs_time.png",
    "keybuffers_stats.png",
    "outages_vs_time.png",
    "rounds_vs_time.png",
]

cols = st.columns(2)
shown = 0
for i, name in enumerate(baseline_imgs):
    path = BASELINE / name
    if path.exists():
        cols[i % 2].image(str(path), caption=name, use_container_width=True)
        shown += 1

if shown == 0:
    st.info("No baseline images found in `outputs/baseline/`.")

st.subheader("Sweeps / heatmaps")
plot_choices = []
for folder in [SWEEP, PLOTS_COMPARE]:
    if folder.exists():
        for p in sorted(folder.glob("*.png")):
            plot_choices.append(p)

if plot_choices:
    selection = st.selectbox(
        "Select a plot",
        plot_choices,
        format_func=lambda p: p.name,
    )
    st.image(str(selection), use_container_width=True)
else:
    st.info("No sweep plots found in `outputs/sweep/` or `plots_compare_arrangements/`.")

csv_path = SWEEP / "sweep_results.csv"
if csv_path.exists():
    st.subheader("Sweep table")
    df = pd.read_csv(csv_path)
    st.dataframe(df, use_container_width=True)
