import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from cislunar_qfl.app.sim_api import LiveRunConfig, SweepConfig, run_sweep

st.title("Trade Study / Sweeps")
st.caption("Generate trade surfaces on demand (no precomputed files required).")

with st.sidebar:
    st.subheader("Sweep ranges")
    clients_min = st.slider("Clients min", 1, 64, 4, 1)
    clients_max = st.slider("Clients max", 1, 64, 32, 1)
    clients_step = st.selectbox("Clients step", [1, 2, 4, 8], index=2)

    update_min = st.selectbox("Update min (bits)", [2e5, 5e5, 1e6], index=0)
    update_max = st.selectbox("Update max (bits)", [2e6, 5e6, 1e7], index=1)
    update_steps = st.selectbox("Update steps", [3, 5, 7, 9], index=1)

    st.subheader("Base assumptions")
    round_period_s = st.slider("Round period (s)", 5, 600, 30, 5)
    sim_hours = st.slider("Sim horizon (hours)", 0.1, 2.0, 0.5, 0.1)
    target_accuracy = st.slider("Target accuracy", 0.60, 0.99, 0.90, 0.01)
    mode = st.radio("Mode", ["qkd_keying", "otp"], horizontal=True)
    session_key_bits = st.selectbox("Session key (bits)", [128, 256, 512, 1024], index=1)
    key_buffer_capacity_bits = st.selectbox(
        "Key buffer capacity (bits)", [5e5, 1e6, 2e6, 5e6, 1e7], index=2
    )
    qkd_throughput_bps = st.selectbox("QKD throughput (bps)", [5e3, 1e4, 2e4, 5e4], index=2)
    contact_window_s = st.slider("Contact window (s)", 60, 3600, 600, 60)
    contact_gap_s = st.slider("Contact gap (s)", 60, 3600, 1200, 60)
    seed = st.number_input("Seed", 0, 10000, 0)


def _build_clients(min_v: int, max_v: int, step: int) -> np.ndarray:
    if max_v < min_v:
        min_v, max_v = max_v, min_v
    return np.arange(min_v, max_v + 1, step, dtype=int)


def _build_updates(min_v: float, max_v: float, steps: int) -> np.ndarray:
    lo = float(min_v)
    hi = float(max_v)
    if hi < lo:
        lo, hi = hi, lo
    return np.linspace(lo, hi, steps, dtype=float)


def _heatmap(fig_title: str, values: np.ndarray, clients: np.ndarray, updates: np.ndarray):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    im = ax.imshow(values, origin="lower", aspect="auto")
    ax.set_title(fig_title)
    ax.set_xlabel("Update size (bits)")
    ax.set_ylabel("Clients per round")
    ax.set_xticks(np.arange(len(updates)))
    ax.set_xticklabels([f"{int(u):,}" for u in updates], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(clients)))
    ax.set_yticklabels([str(c) for c in clients])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


if st.button("Run sweep"):
    clients = _build_clients(clients_min, clients_max, clients_step)
    updates = _build_updates(update_min, update_max, update_steps)

    base_cfg = LiveRunConfig(
        clients_per_round=int(clients[0]),
        round_period_s=int(round_period_s),
        update_size_bits=float(updates[0]),
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

    sweep_cfg = SweepConfig(
        clients_list=tuple(int(c) for c in clients),
        update_bits_list=tuple(float(u) for u in updates),
        base_cfg=base_cfg,
    )

    @st.cache_data(show_spinner=False)
    def _cached_sweep(cfg: SweepConfig):
        return run_sweep(cfg)

    results = _cached_sweep(sweep_cfg)

    st.subheader("Heatmaps")
    st.pyplot(_heatmap("Time to target (s)", results["time_to_target"], clients, updates))
    st.pyplot(_heatmap("Outage rate", results["outage_rate"], clients, updates))
    st.pyplot(_heatmap("Final accuracy", results["final_accuracy"], clients, updates))

    st.subheader("Sweep table")
    rows = []
    for i, c in enumerate(clients):
        for j, u in enumerate(updates):
            rows.append(
                {
                    "clients_per_round": int(c),
                    "update_size_bits": float(u),
                    "time_to_target_s": results["time_to_target"][i, j],
                    "outage_rate": results["outage_rate"][i, j],
                    "final_accuracy": results["final_accuracy"][i, j],
                }
            )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download sweep results (CSV)",
        csv_bytes,
        file_name="fl_sweep_results.csv",
        mime="text/csv",
    )
