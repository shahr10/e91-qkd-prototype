from __future__ import annotations

import pandas as pd
import streamlit as st

from cislunar_qfl.app.sim_api import LiveRunConfig, simulate

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
    seed = st.number_input("Seed", 0, 10000, 0)


st.subheader("Live Simulation")
st.caption("Generated on demand using an assumed backend model (fast, deterministic).")
with st.expander("Assumptions (editable via inputs)", expanded=False):
    st.markdown(
        \"\"\"\n- Accuracy improves only on successful (non-outage) rounds.\n- Learning rate scales with `clients_per_round` (diminishing returns).\n- Larger updates can improve learning but increase key demand.\n- Key supply is periodic: contact window + gap; buffer caps at capacity.\n\"\"\"\n    )

if st.button("Run live simulation"):
    cfg = LiveRunConfig(
        clients_per_round=int(clients_per_round),
        round_period_s=int(round_period_s),
        update_size_bits=float(update_size_bits),
        target_accuracy=float(target_accuracy),
        mode=str(mode),
        session_key_bits=int(session_key_bits),
        key_buffer_capacity_bits=float(key_buffer_capacity_bits),
        qkd_throughput_bps=float(qkd_throughput_bps),
        contact_window_s=int(contact_window_s),
        contact_gap_s=int(contact_gap_s),
        sim_hours=float(sim_hours),
        seed=int(seed),
    )

    @st.cache_data(show_spinner=False)
    def _cached_run(cfg: LiveRunConfig):
        return simulate(cfg)

    results = _cached_run(cfg)

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

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download live run (CSV)",
        csv_bytes,
        file_name="fl_live_run.csv",
        mime="text/csv",
    )
