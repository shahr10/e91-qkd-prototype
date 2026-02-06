from __future__ import annotations

import json
import io
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from cislunar_constellation.config import ConstellationConfig
from cislunar_constellation.engine.orbits import VALIDATED_ORBITS
from cislunar_constellation.run import run_design

st.title("Cislunar Constellation Designer")
st.caption("Engine-backed optimization for cislunar quantum relay constellations.")

# Sidebar controls
with st.sidebar:
    st.subheader("Presets")
    preset = st.selectbox(
        "Scenario presets",
        ["Default", "High Coverage", "Low Satellites", "High Key Rate"],
        index=0,
    )

    st.subheader("Mode")
    mode = st.radio("Interface mode", ["Basic", "Advanced"], horizontal=True)

    st.subheader("Scenario")
    horizon_hours = st.slider("Time horizon (hours)", 4.0, 72.0, 24.0, 2.0)
    n_time = st.slider("Time samples", 12, 144, 36, 6)

    st.subheader("Orbits")
    families = sorted({v["family"] for v in VALIDATED_ORBITS.values()})
    if preset == "High Coverage":
        default_families = ["Halo", "DRO", "Lyapunov", "Vertical"]
        max_candidates = 40
    elif preset == "Low Satellites":
        default_families = ["Halo", "Lyapunov"]
        max_candidates = 15
    elif preset == "High Key Rate":
        default_families = ["DRO", "Halo"]
        max_candidates = 30
    else:
        default_families = ["Halo", "DRO", "Lyapunov"]
        max_candidates = 20

    selected_families = st.multiselect(
        "Orbit families",
        families,
        default=[f for f in default_families if f in families],
    )
    max_candidates = st.slider("Max candidates", 5, 80, max_candidates, 5)

    st.subheader("Optimization")
    if preset == "Low Satellites":
        target_default = 8
    elif preset == "High Coverage":
        target_default = 16
    elif preset == "High Key Rate":
        target_default = 14
    else:
        target_default = 12
    target_sats = st.slider("Target satellites", 4, 40, target_default, 1)

    if mode == "Advanced":
        st.subheader("GA Settings")
        ngen = st.slider("Generations", 5, 100, 25, 5)
        pop_size = st.slider("Population", 10, 200, 50, 10)
        cxpb = st.slider("Crossover prob", 0.1, 1.0, 0.7, 0.05)
        mutpb = st.slider("Mutation prob", 0.05, 0.8, 0.2, 0.05)

        st.subheader("Network")
        allow_isl = st.checkbox("Allow ISL", value=True)
        max_isl = st.slider("Max ISL partners", 0, 6, 3, 1)
        isl_dist = st.slider("ISL max distance (km)", 50000, 300000, 150000, 10000)

        st.subheader("Ground")
        elevation_mask = st.slider("Elevation mask (deg)", 0.0, 40.0, 10.0, 1.0)
        availability = st.slider("Availability", 0.1, 1.0, 1.0, 0.05)
    else:
        ngen = 25
        pop_size = 50
        cxpb = 0.7
        mutpb = 0.2
        allow_isl = True
        max_isl = 3
        isl_dist = 150000
        elevation_mask = 10.0
        availability = 1.0

cfg = ConstellationConfig()
cfg.time.horizon_hours = float(horizon_hours)
cfg.time.n_time = int(n_time)
cfg.orbits.families = selected_families or None
cfg.orbits.max_candidates = int(max_candidates)
cfg.optimize.target_sats = int(target_sats)
cfg.optimize.ngen = int(ngen)
cfg.optimize.pop_size = int(pop_size)
cfg.optimize.cxpb = float(cxpb)
cfg.optimize.mutpb = float(mutpb)
cfg.network.allow_isl = bool(allow_isl)
cfg.network.max_isl_partners = int(max_isl)
cfg.network.isl_distance_km = float(isl_dist)
cfg.ground.elevation_mask_deg = float(elevation_mask)
cfg.ground.availability = float(availability)

for warn in cfg.validate():
    st.warning(warn)

st.markdown("---")

if st.button("Run Optimization", type="primary"):
    progress = st.progress(0, text="Propagating orbits...")

    def _progress_cb(stage: str, value: float):
        label_map = {
            "propagate": "Propagating orbits...",
            "visibility": "Computing visibility...",
            "ga": "Running genetic algorithm...",
            "roles": "Assigning roles...",
            "done": "Finalizing...",
        }
        progress.progress(int(value * 100), text=label_map.get(stage, stage))

    with st.spinner("Running design optimization..."):
        results = run_design(cfg, progress_cb=_progress_cb)
    progress.empty()

    st.success("Optimization complete.")

    st.subheader("Key Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Coverage", f"{results.metrics['coverage']:.1%}")
    c2.metric("Key Rate (bps)", f"{results.metrics['rate']:.2e}")
    c3.metric("Diversity", f"{results.metrics['diversity']}")

    st.subheader("Selected Orbits")
    st.write(results.selected_orbits)

    st.subheader("Orbit Families (Selected)")
    family_counts = {}
    for name in results.selected_orbits:
        fam = VALIDATED_ORBITS[name]["family"]
        family_counts[fam] = family_counts.get(fam, 0) + 1
    if family_counts:
        st.bar_chart(family_counts)

    st.subheader("Roles")
    st.write(results.roles)

    st.subheader("Coverage Heatmap (Time × Ground Station)")
    # Build time × GS coverage for selected orbits
    if results.selected_orbits:
        sel_idx = [results.orbit_names.index(s) for s in results.selected_orbits if s in results.orbit_names]
        if sel_idx:
            coverage = np.max(results.vis_M[sel_idx, :, :], axis=0)  # shape (time, gs)

            fig, ax = plt.subplots(figsize=(8, 3.8))
            im = ax.imshow(coverage.T, aspect="auto", origin="lower", cmap="viridis")
            ax.set_xlabel("Time index")
            ax.set_ylabel("Ground station")
            ax.set_yticks(range(len(cfg.ground.stations)))
            ax.set_yticklabels([gs.name for gs in cfg.ground.stations])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Visibility (0/1)")
            st.pyplot(fig, width="stretch")

            buf_png = io.BytesIO()
            fig.savefig(buf_png, format="png", dpi=200, bbox_inches="tight")
            buf_png.seek(0)

            buf_pdf = io.BytesIO()
            fig.savefig(buf_pdf, format="pdf", bbox_inches="tight")
            buf_pdf.seek(0)

            st.download_button(
                "Download Heatmap (PNG)",
                buf_png,
                file_name="coverage_heatmap.png",
                mime="image/png",
            )
            st.download_button(
                "Download Heatmap (PDF)",
                buf_pdf,
                file_name="coverage_heatmap.pdf",
                mime="application/pdf",
            )

    export = {
        "config": cfg.__dict__,
        "selected_orbits": results.selected_orbits,
        "metrics": {k: v for k, v in results.metrics.items() if k != "logbook"},
        "roles": results.roles,
        "downlink_map": results.downlink_map,
        "isl_partners": results.isl_partners,
    }

    st.download_button(
        "Download Results (JSON)",
        json.dumps(export, default=str, indent=2),
        file_name="constellation_results.json",
        mime="application/json",
    )
