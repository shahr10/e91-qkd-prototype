from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
BASELINE = OUTPUTS / "baseline"
SWEEP = OUTPUTS / "sweep"
PLOTS_COMPARE = ROOT / "plots_compare_arrangements"

st.title("Federated Learning in Space")

st.markdown(
    """
This page answers three questions:

- What is training in space? (rounds, clients, update size, target accuracy)
- What limits it? (contact windows, RF/QKD throughput, key-buffer constraints, outages)
- What's optimal? (trade plots: clients/round vs time-to-target vs key failures)
"""
)

with st.sidebar:
    st.subheader("Live Run Inputs")
    clients_per_round = st.slider("Clients per round", 1, 64, 8, 1)
    round_period_s = st.slider("Round period (s)", 5, 600, 30, 5)
    update_size_bits = st.selectbox(
        "Update size (bits)",
        [2e5, 5e5, 1e6, 2e6, 5e6],
        index=3,
        format_func=lambda v: f"{int(v):,}",
    )
    target_accuracy = st.slider("Target accuracy", 0.60, 0.99, 0.90, 0.01)

    st.subheader("Keying Mode")
    mode = st.radio("Mode", ["qkd_keying", "otp"], horizontal=True)
    session_key_bits = st.selectbox(
        "Session key size (bits)",
        [128, 256, 512, 1024],
        index=1,
        help="Used for QKD-keying mode. OTP ignores this.",
    )
    key_buffer_capacity_bits = st.selectbox(
        "Key buffer capacity (bits)",
        [5e5, 1e6, 2e6, 5e6, 1e7],
        index=2,
        format_func=lambda v: f"{int(v):,}",
    )

    st.subheader("Contact / Throughput")
    qkd_throughput_bps = st.selectbox(
        "QKD throughput (bps)",
        [5e3, 1e4, 2e4, 5e4, 1e5],
        index=2,
        format_func=lambda v: f"{int(v):,}",
    )
    contact_window_s = st.slider("Contact window (s)", 60, 3600, 600, 60)
    contact_gap_s = st.slider("Contact gap (s)", 60, 3600, 1200, 60)
    sim_hours = st.slider("Sim horizon (hours)", 0.1, 2.0, 0.5, 0.1)


def _simulate_live_run() -> dict[str, object]:
    total_seconds = int(sim_hours * 3600)
    total_rounds = max(1, total_seconds // round_period_s)

    rounds = np.arange(total_rounds)
    time_s = rounds * round_period_s

    # Simple accuracy curve (placeholder, replace with real backend)
    growth_scale = max(1.0, total_rounds / 6.0)
    accuracy = 1.0 - np.exp(-rounds / growth_scale)
    accuracy = 0.5 + 0.5 * accuracy  # map to [0.5, 1.0)

    # Key buffer dynamics (placeholder model)
    key_demand_bits = update_size_bits if mode == "otp" else session_key_bits

    buffer = np.zeros_like(time_s, dtype=float)
    outages = np.zeros_like(time_s, dtype=int)

    cycle = contact_window_s + contact_gap_s
    for i in range(total_rounds):
        t = time_s[i]
        in_contact = (t % cycle) < contact_window_s
        supply = qkd_throughput_bps * (round_period_s if in_contact else 0)

        prev = buffer[i - 1] if i > 0 else key_buffer_capacity_bits / 2
        available = min(key_buffer_capacity_bits, prev + supply)

        if available >= key_demand_bits:
            buffer[i] = available - key_demand_bits
        else:
            buffer[i] = available
            outages[i] = 1

    time_to_target = None
    idx = np.where(accuracy >= target_accuracy)[0]
    if idx.size > 0:
        time_to_target = time_s[idx[0]]

    return {
        "time_s": time_s,
        "rounds": rounds,
        "accuracy": accuracy,
        "buffer": buffer,
        "outages": outages,
        "time_to_target": time_to_target,
        "key_demand_bits": key_demand_bits,
    }


st.subheader("Live Simulation")
st.caption("Placeholder dynamics for the UI. Swap in the real backend when available.")

if st.button("Run live simulation"):
    results = _simulate_live_run()

    col1, col2, col3 = st.columns(3)
    col1.metric("Clients per round", f"{clients_per_round}")
    col2.metric("Key demand (bits/update)", f"{int(results['key_demand_bits']):,}")
    if results["time_to_target"] is None:
        col3.metric("Time to target", "Not reached")
    else:
        col3.metric("Time to target", f"{results['time_to_target'] / 60:.1f} min")

    df = pd.DataFrame(
        {
            "time_s": results["time_s"],
            "accuracy": results["accuracy"],
            "key_buffer_bits": results["buffer"],
            "outage": results["outages"],
        }
    )

    st.markdown("**Accuracy vs time**")
    st.line_chart(df.set_index("time_s")["accuracy"])

    st.markdown("**Key buffer vs time**")
    st.line_chart(df.set_index("time_s")["key_buffer_bits"])

    outage_rate = float(df["outage"].mean())
    st.markdown("**Outage / update rejection rate**")
    st.write(f"{outage_rate:.2%} of rounds")

    buffer_stats = df["key_buffer_bits"].quantile([0.1, 0.5, 0.9]).to_dict()
    st.markdown("**Key buffer summary**")
    st.write(
        {
            "p10": int(buffer_stats[0.1]),
            "median": int(buffer_stats[0.5]),
            "p90": int(buffer_stats[0.9]),
        }
    )


st.subheader("Precomputed Results (if present)")

baseline_imgs = [
    "accuracy_vs_time.png",
    "keybuffers_stats.png",
    "outages_vs_time.png",
    "rounds_vs_time.png",
]

found_any = False
cols = st.columns(2)
for i, name in enumerate(baseline_imgs):
    path = BASELINE / name
    if path.exists():
        cols[i % 2].image(str(path), caption=f"Baseline: {name}", use_container_width=True)
        found_any = True

if not found_any:
    st.info(
        "No baseline images found. Place them in `outputs/baseline/` to show them here."
    )

st.markdown("**Sweep/Trade plots**")

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
    st.info(
        "No sweep plots found. Add PNGs to `outputs/sweep/` or `plots_compare_arrangements/`."
    )

csv_path = SWEEP / "sweep_results.csv"
if csv_path.exists():
    st.markdown("**Sweep results table**")
    df = pd.read_csv(csv_path)
    st.dataframe(df, use_container_width=True)
