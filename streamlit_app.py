import streamlit as st

st.set_page_config(
    page_title="E91 QKD + Federated Learning",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("E91 QKD Testbench + Federated Learning in Space")

st.markdown(
    """
**Simulate QKD link performance + key buffer dynamics for space networks.**

**What you can do here**
- Choose a scenario
- Run the simulation
- Download plots and data

**Credibility**
- Assumed model with explicit inputs shown in the sidebar
- Reproducible outputs (CSV downloads)
- Updated: February 5, 2026

This site tells one tight story:

- What is training in space? (rounds, clients, update size, target accuracy)
- What limits it? (contact windows, RF/QKD throughput, key-buffer constraints, outages)
- What's optimal? (trade plots: clients/round vs time-to-target vs key failures)

Use the left sidebar to navigate:

- **QKD Prototype** for the E91 simulator
- **In-Space Federated Learning** for the live demo and story framing
- **Trade Study** for precomputed sweeps and heatmaps
"""
)

st.info(
    "Tip: In Streamlit Cloud, set the main file to `streamlit_app.py` to enable multipage navigation."
)
