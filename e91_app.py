#!/usr/bin/env python3
"""
================================================================================
E91 QKD Professional Interface (v7.0 Modular)
================================================================================

Professional quantum key distribution simulator with modern, user-friendly design.

NEW IN v7.0 Modular:
- 🎨 Professional dark theme with modern styling
- 🔧 Dynamic presets that update ALL parameter values
- 🎲 Smart randomizers that update displayed inputs
- 💡 Clean tooltips (no info dropdowns)
- ⚡ Streamlined, responsive layout
- 🌐 Enhanced two-way communication
- ✅ Fixed all deprecation warnings
- 📦 Modular architecture for easier debugging

Run with:   cd PrototypeSetup
            streamlit run e91_app.py

================================================================================
"""

# ============================================================================
# IMPORTS - External Libraries
# ============================================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import time
import logging
import io
from typing import Dict
from dataclasses import asdict

# ============================================================================
# IMPORTS - Our Modular Components
# ============================================================================

# Data structures
from ApplicationStructure.models import ExperimentConfig, ExperimentResults, MessageTest, MessageListener

# Styling
from ApplicationStructure.styles import PROFESSIONAL_CSS, BASE_CSS

# Core functionality
from ApplicationStructure.quantum_protocol import E91Protocol, QISKIT_AVAILABLE, QUTIP_AVAILABLE
from ApplicationStructure.encryption import (message_to_bits, bits_to_message, xor_encrypt_decrypt,
                       test_self_message, test_two_party_message, listener_receive_message)
from ApplicationStructure.communication import send_udp_message, receive_udp_message

# Analysis and visualization
from ApplicationStructure.analysis import run_parameter_sweep, create_sweep_plots

# Utilities
from ApplicationStructure.utils import (normalize_probs, format_hex, history_empty_card, h,
                  compute_satellite_loss, estimate_runtime, validate_config,
                  create_angle_preview, create_loss_budget_preview)

# Dependency checking
from ApplicationStructure.dependency_check import (check_backend_dependencies, get_install_commands,
                                get_missing_dependencies, validate_backend_selection,
                                get_available_backends)

# Presets
from ApplicationStructure.presets import get_preset_config, PRESET_NAMES

# UI Helpers
from ApplicationStructure.ui_helpers import (apply_preset_to_session_state, build_experiment_config,
                         display_experiment_results)

# Configuration constants (optional, with fallbacks)
try:
    from ApplicationStructure.config import (
        QuantumConstants, MeasurementAngles, CoreParameters,
        NoiseParameters, ChannelLossParameters, DetectorParameters,
        SourceParameters, SatelliteParameters, SecurityParameters,
        DeviceIndependentParameters, BackgroundParameters, EavesdropperParameters,
        NetworkParameters, ValidationRanges
    )
    VERSION = "7.0.0 Professional Modular (with config.py)"
    CHSH_CLASSICAL_BOUND = QuantumConstants.CHSH_CLASSICAL_BOUND
    CHSH_QUANTUM_MAX = QuantumConstants.CHSH_QUANTUM_MAX
    QBER_THRESHOLD = QuantumConstants.QBER_THRESHOLD
    ALICE_ANGLES_DEFAULT = MeasurementAngles.ALICE_ANGLES_DEFAULT
    BOB_ANGLES_DEFAULT = MeasurementAngles.BOB_ANGLES_DEFAULT
    CONFIG_LOADED = True
except ImportError:
    VERSION = "7.0.0 Professional Modular"
    CHSH_CLASSICAL_BOUND = 2.0
    CHSH_QUANTUM_MAX = 2.0 * np.sqrt(2)
    QBER_THRESHOLD = 0.11
    ALICE_ANGLES_DEFAULT = [0.0, np.pi/8, np.pi/4]
    BOB_ANGLES_DEFAULT = [np.pi/8, np.pi/4, 3*np.pi/8]
    CONFIG_LOADED = False
    # Create fallback classes with basic defaults
    class CoreParameters:
        NUM_PAIRS_MIN = 100
        NUM_PAIRS_MAX = 10000000
    class NoiseParameters:
        DEPOLARIZING_ALPHA_MIN = 0.0
        DEPOLARIZING_ALPHA_MAX = 1.0
        VISIBILITY_MAX = 1.0
        INTRINSIC_ERROR_ED_MIN = 0.0
        INTRINSIC_ERROR_ED_MAX = 0.5
    class ChannelLossParameters:
        DISTANCE_KM_A_MIN = 0.0
        DISTANCE_KM_B_MIN = 0.0
        FIBER_LOSS_DB_PER_KM_DEFAULT = 0.2
    class DetectorParameters:
        HERALDING_EFFICIENCY_MAX = 1.0
        DARK_PROB_MIN = 0.0
        DARK_PROB_MAX = 0.1
        JITTER_NS_MIN = 0.0
        JITTER_NS_MAX = 10.0
    class SourceParameters:
        SPDC_BRIGHTNESS_MU_MIN = 0.001
        SPDC_BRIGHTNESS_MU_DEFAULT = 0.1
    class ValidationRanges:
        ANGLE_MIN = 0.0
        ANGLE_MAX = 2 * np.pi
    class NetworkParameters:
        PORT_MIN = 1024
        PORT_MAX = 65535
        DEFAULT_LOCAL_PORT = 8765
        DEFAULT_PEER_PORT = 8766
        DEFAULT_LOCALHOST = "127.0.0.1"
        DEFAULT_PEER_HOST = "127.0.0.1"
        AUTO_REFRESH_INTERVAL_SEC = 2.0
        PROBABILITY_MIN = 0.0
        PROBABILITY_MAX = 1.0

# Quantum math functions (optional, with fallbacks)
try:
    from ApplicationStructure.quantum_math import (binary_entropy, mutual_information, compute_correlation,
                             compute_chsh_parameter, theoretical_chsh_value,
                             secret_key_rate_asymptotic, secret_key_rate_finite,
                             di_key_rate_lower_bound, normalize_probabilities)
    MATH_LOADED = True
except ImportError:
    MATH_LOADED = False
    # Basic fallback
    def normalize_probabilities(probs):
        return normalize_probs(probs)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="E91 QKD Professional",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

# Sidebar flow guide
st.sidebar.subheader("Flow")
st.sidebar.markdown(
    """
1. Scenario
2. QKD model
3. Demand model
4. Run
5. Results
"""
)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'config' not in st.session_state:
    st.session_state.config = ExperimentConfig()

if 'results' not in st.session_state:
    st.session_state.results = None

if 'experiment_history' not in st.session_state:
    st.session_state.experiment_history = []

if 'sweep_history' not in st.session_state:
    st.session_state.sweep_history = []

if 'message_history' not in st.session_state:
    st.session_state.message_history = []

if 'listener' not in st.session_state:
    st.session_state.listener = MessageListener()

if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Mode toggle for progressive disclosure
if 'ui_mode' not in st.session_state:
    st.session_state.ui_mode = 'Basic'

st.sidebar.subheader("Mode")
st.session_state.ui_mode = st.sidebar.radio(
    "Interface mode",
    ["Basic", "Advanced"],
    index=0 if st.session_state.ui_mode == "Basic" else 1,
    horizontal=True,
)

# Initialize preset state
if 'current_preset' not in st.session_state:
    st.session_state.current_preset = "Custom"

# Callback to switch to Custom when user modifies inputs
def switch_to_custom():
    """Switch to Custom preset when user modifies any input."""
    if st.session_state.current_preset != "Custom":
        st.session_state.current_preset = "Custom"

# Initialize all checkbox states for losses/noise
if 'enable_depol' not in st.session_state:
    st.session_state.enable_depol = False
if 'depol_alpha' not in st.session_state:
    st.session_state.depol_alpha = 0.0
if 'enable_vis' not in st.session_state:
    st.session_state.enable_vis = False
if 'visibility' not in st.session_state:
    st.session_state.visibility = 1.0
if 'enable_intrinsic' not in st.session_state:
    st.session_state.enable_intrinsic = False
if 'intrinsic_ed' not in st.session_state:
    st.session_state.intrinsic_ed = 0.0
if 'enable_fiber' not in st.session_state:
    st.session_state.enable_fiber = False
if 'enable_insertion' not in st.session_state:
    st.session_state.enable_insertion = False
if 'enable_misalignment' not in st.session_state:
    st.session_state.enable_misalignment = False
if 'enable_jitter' not in st.session_state:
    st.session_state.enable_jitter = False
if 'enable_deadtime' not in st.session_state:
    st.session_state.enable_deadtime = False
if 'enable_after' not in st.session_state:
    st.session_state.enable_after = False
if 'enable_sat' not in st.session_state:
    st.session_state.enable_sat = False
if 'enable_bg' not in st.session_state:
    st.session_state.enable_bg = False
if 'enable_multi' not in st.session_state:
    st.session_state.enable_multi = False
if 'enable_satellite' not in st.session_state:
    st.session_state.enable_satellite = False
if 'enable_eve' not in st.session_state:
    st.session_state.enable_eve = False
if 'enable_repeater' not in st.session_state:
    st.session_state.enable_repeater = False
if 'enable_detector' not in st.session_state:
    st.session_state.enable_detector = False
if 'herald_eff' not in st.session_state:
    st.session_state.herald_eff = 0.9
if 'end_det_eff' not in st.session_state:
    st.session_state.end_det_eff = 0.8
if 'enable_dark' not in st.session_state:
    st.session_state.enable_dark = False

# Initialize seed
if "seed" not in st.session_state:
    st.session_state["seed"] = 42

# Initialize distances
if 'distance_A' not in st.session_state:
    st.session_state['distance_A'] = 5.0
if 'distance_B' not in st.session_state:
    st.session_state['distance_B'] = 5.0

# Initialize repeater parameters
if 'num_repeaters' not in st.session_state:
    st.session_state['num_repeaters'] = 2
if 'repeater_gain_dB' not in st.session_state:
    st.session_state['repeater_gain_dB'] = 3.0

# Initialize angles
for key in ["alice_0", "alice_1", "alice_2", "bob_0", "bob_1", "bob_2"]:
    if key not in st.session_state:
        default_val = {'alice_0': 0.0, 'alice_1': np.pi/8, 'alice_2': np.pi/4,
                      'bob_0': np.pi/8, 'bob_1': np.pi/4, 'bob_2': 3*np.pi/8}
        st.session_state[key] = default_val[key]

# Initialize basis probabilities
if 'pA' not in st.session_state:
    st.session_state['pA'] = [0.5, 0.5, 0.0]
if 'pB' not in st.session_state:
    st.session_state['pB'] = [0.5, 0.5, 0.0]

# Initialize num_pairs
if 'num_pairs' not in st.session_state:
    st.session_state['num_pairs'] = 10000

# Initialize fiber loss parameters
if 'fiber_loss_db' not in st.session_state:
    st.session_state['fiber_loss_db'] = 0.2
if 'loss_a' not in st.session_state:
    st.session_state['loss_a'] = 0.0
if 'loss_b' not in st.session_state:
    st.session_state['loss_b'] = 0.0

# Initialize dark count parameters
if 'use_cps' not in st.session_state:
    st.session_state['use_cps'] = False
if 'dark_cps' not in st.session_state:
    st.session_state['dark_cps'] = 100.0
if 'dark_prob' not in st.session_state:
    st.session_state['dark_prob'] = 0.001

# Initialize timing parameters
if 'jitter_ns' not in st.session_state:
    st.session_state['jitter_ns'] = 0.5
if 'deadtime_ns' not in st.session_state:
    st.session_state['deadtime_ns'] = 50.0
if 'after_prob' not in st.session_state:
    st.session_state['after_prob'] = 0.05
if 'after_delay' not in st.session_state:
    st.session_state['after_delay'] = 100.0
if 'saturation_rate' not in st.session_state:
    st.session_state['saturation_rate'] = 1e6

# Initialize background/coincidence parameters
if 'Y0' not in st.session_state:
    st.session_state['Y0'] = 1e-6
if 'bg_cps' not in st.session_state:
    st.session_state['bg_cps'] = 500.0
if 'coinc_ns' not in st.session_state:
    st.session_state['coinc_ns'] = 1.0
if 'rep_rate' not in st.session_state:
    st.session_state['rep_rate'] = 1e6
if 'enable_timetag' not in st.session_state:
    st.session_state['enable_timetag'] = False

# Initialize multi-pair (SPDC) parameters
if 'spdc_mu' not in st.session_state:
    st.session_state['spdc_mu'] = 0.1
if 'pair_rate' not in st.session_state:
    st.session_state['pair_rate'] = 1e6
if 'pump_mw' not in st.session_state:
    st.session_state['pump_mw'] = 10.0
if 'wavelength' not in st.session_state:
    st.session_state['wavelength'] = 810.0
if 'filt_band' not in st.session_state:
    st.session_state['filt_band'] = 3.0

# Initialize satellite parameters
if 'beam_div' not in st.session_state:
    st.session_state['beam_div'] = 10.0
if 'pointing' not in st.session_state:
    st.session_state['pointing'] = 5.0
if 'rx_fov' not in st.session_state:
    st.session_state['rx_fov'] = 100.0
if 'slant_km' not in st.session_state:
    st.session_state['slant_km'] = 500.0
if 'tx_ap' not in st.session_state:
    st.session_state['tx_ap'] = 0.3
if 'rx_ap' not in st.session_state:
    st.session_state['rx_ap'] = 1.0
if 'is_day' not in st.session_state:
    st.session_state['is_day'] = False
if 'sat_bg_cps' not in st.session_state:
    st.session_state['sat_bg_cps'] = 1000.0
if 'enable_turb' not in st.session_state:
    st.session_state['enable_turb'] = False
if 'cn2' not in st.session_state:
    st.session_state['cn2'] = 1e-15

# Initialize eavesdropper parameters
if 'eve_model' not in st.session_state:
    st.session_state['eve_model'] = "intercept_resend"
if 'eve_prob' not in st.session_state:
    st.session_state['eve_prob'] = 0.0


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main Streamlit application entry point."""

    # Top-left action buttons + centered title
    # Responsive header layout
    header_col1, header_col2, header_col3 = st.columns([1, 10, 1])
    with header_col1:
        guide_clicked = st.button("📖", help="Open application guide", key="guide_btn", use_container_width=False)
    with header_col2:
        st.markdown(
            f"""
            <div style='text-align:center;'>
              <h1 style='margin-bottom:0; font-size:2.5rem;'>E91 QKD Enhanced Testbench</h1>
            </div>
            <hr style='margin-top:10px;'/>
            """,
            unsafe_allow_html=True,
        )
    with header_col3:
        st.write("")  # Spacer for balance

    if guide_clicked:
        with st.expander("Application Guide", expanded=True):
            st.markdown(
                """
                **Overview**
                - **Experiment tab**: configure angles, probabilities, losses/noise, security, DI, and Eve. Run to produce key material and plots.
                - **Parametric Sweep**: vary one parameter across a range and visualize S, QBER, key rates, and detection.
                - **Message Test**: encrypt a message using OTP with the sifted key (self-test or two-party).
                - **History**: browse, clear, and download logs of experiments, sweeps, and messages.

                **Graphs (Experiment)**
                - Bell Test Result: bar chart of measured CHSH S vs classical (2.0) and quantum (2.828) limits.
                - Quantum Bit Error Rate: bar of QBER with 11% dashed threshold.
                - Key Generation Pipeline: generated→detected→sifted→final.
                - Secret Fractions: asymptotic and finite-key secret fraction per sifted bit.
                - Key Rates: secret bits per generated pair.
                - Correlators Heatmap: E(a_i,b_j) across bases; structure indicates CHSH geometry.
                - Angles: polar preview of Alice/Bob bases.

                **Results/Terms**
                - Detection eta: detected/generated fraction; dominated by loss and efficiencies.
                - Sifting eta: sifted/detected fraction; ~22% ideal for E91 with uniform bases.
                - QBER: fraction of mismatched bits in the sifted key; <11% for security.
                - Secret fraction: upper bound on extractable secret bits per sifted bit.
                - Final key bits: after IR and privacy amplification.

                **Security**
                - Secure if S>2 and QBER<11% (heuristic; details depend on assumptions and finite-size effects).

                **Message Testing**
                - One-time pad XOR using the sifted key. Two-party mode exposes mismatches due to QBER.
                """
            )

    # ========================================================================
    # DEPENDENCY CHECK AND INSTALL GUIDANCE
    # ========================================================================

    deps = check_backend_dependencies()
    missing = get_missing_dependencies(deps)
    install_cmds = get_install_commands()

    # Show dependency status in an expander
    with st.expander("🔧 Backend Dependencies Status", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Qiskit**")
            if deps['qiskit'] and deps['qiskit_aer']:
                st.success("✅ Installed")
            else:
                st.error("❌ Not installed")

        with col2:
            st.markdown("**Qiskit Aer**")
            if deps['qiskit_aer']:
                st.success("✅ Installed")
            else:
                st.error("❌ Not installed")

        with col3:
            st.markdown("**QuTiP**")
            if deps['qutip']:
                st.success("✅ Installed")
            else:
                st.error("❌ Not installed")

        # If any dependencies missing, show install commands
        if missing:
            st.markdown("---")
            st.markdown("### 📦 Installation Commands")

            # Offer to install both backends
            if len(missing) == 2:
                st.info("**Recommended**: Install both backends for maximum compatibility")
                st.markdown("**Using pip:**")
                st.code(install_cmds['both']['pip'], language='bash')
                st.markdown("**Using conda:**")
                st.code(install_cmds['both']['conda'], language='bash')
            else:
                # Install only missing backend
                for backend_name in missing:
                    st.markdown(f"**Install {backend_name.capitalize()}:**")
                    st.markdown("**Using pip:**")
                    st.code(install_cmds[backend_name]['pip'], language='bash')
                    st.markdown("**Using conda:**")
                    st.code(install_cmds[backend_name]['conda'], language='bash')

            st.warning("⚠️ After installing dependencies, please **restart the Streamlit app** for changes to take effect.")

        # If only one backend available, offer to install the other
        elif len(missing) == 0 and (not deps['qiskit'] or not deps['qutip']):
            if deps['qiskit'] and deps['qiskit_aer'] and not deps['qutip']:
                st.info("💡 Want to install QuTiP for comparison? See commands above.")
            elif deps['qutip'] and not (deps['qiskit'] and deps['qiskit_aer']):
                st.info("💡 Want to install Qiskit for comparison? See commands above.")

    # Block if NO backends are available
    if not QISKIT_AVAILABLE and not QUTIP_AVAILABLE:
        st.error("❌ **No quantum backend available.** Please install at least one backend (Qiskit or QuTiP) using the commands above.")
        st.stop()

    # Create main tabs for different functionalities
    tab_exp, tab_sweep, tab_msg, tab_hist = st.tabs(["Experiment", "Parametric Sweep", "Message Test", "History"])

    # ========================================================================
    # TAB 1: PARAMETRIC SWEEP (Parameter Analysis)
    # ========================================================================
    # Sweep parameters and analyze QKD performance across ranges
    # ========================================================================
    with tab_sweep:
        st.subheader("Parametric Sweep Analysis")

        # Build base config from session state (doesn't require running experiment first)
        if 'config' in st.session_state and st.session_state.config is not None:
            base_config = st.session_state.config
        else:
            # Use session state to build base config
            # build_experiment_config is already imported at top of file
            try:
                base_config = build_experiment_config(backend)
            except Exception as e:
                st.error(f"Failed to build configuration: {e}")
                st.info("💡 Tip: Configure parameters in the Experiment tab first, then return to run sweeps.")
                base_config = None

        if base_config is not None:
            c1, c2 = st.columns(2)
            with c1:
                # Expanded parameter list with proper names
                param_options = {
                    "Number of Pairs": "num_pairs",
                    "Depolarizing Alpha (α)": "depolarizing_alpha",
                    "Visibility": "visibility",
                    "Intrinsic Error": "intrinsic_error_ed",
                    "Distance (km)": "distance_km",
                    "Fiber Loss (dB/km)": "fiber_loss_dB_per_km",
                    "Heralding Efficiency": "heralding_efficiency",
                    "End Detector Efficiency": "end_detector_efficiency",
                    "Dark Count Probability": "dark_prob",
                    "Timing Jitter (ns)": "jitter_ns",
                    "SPDC Brightness (μ)": "spdc_brightness_mu",
                }

                sweep_param_label = st.selectbox("Parameter to Sweep",
                                          list(param_options.keys()),
                                          help="Select which parameter to vary across a range")
                sweep_param = param_options[sweep_param_label]

                # Set default min/max based on parameter type (using config.py constants)
                if sweep_param == "num_pairs":
                    default_min = float(CoreParameters.NUM_PAIRS_MIN)
                    default_max = min(100000.0, CoreParameters.NUM_PAIRS_MAX)  # Practical limit for sweeps
                elif sweep_param == "depolarizing_alpha":
                    default_min = NoiseParameters.DEPOLARIZING_ALPHA_MIN
                    default_max = NoiseParameters.DEPOLARIZING_ALPHA_MAX
                elif sweep_param == "visibility":
                    default_min = 0.5  # Practical lower bound for QKD
                    default_max = NoiseParameters.VISIBILITY_MAX
                elif sweep_param in ["heralding_efficiency", "end_detector_efficiency"]:
                    default_min = 0.5  # Practical lower bound
                    default_max = DetectorParameters.HERALDING_EFFICIENCY_MAX
                elif sweep_param == "distance_km":
                    default_min = ChannelLossParameters.DISTANCE_KM_A_MIN
                    default_max = 100.0  # Practical limit for fiber QKD
                elif sweep_param == "fiber_loss_dB_per_km":
                    default_min = 0.1  # Practical lower bound
                    default_max = ChannelLossParameters.FIBER_LOSS_DB_PER_KM_DEFAULT * 10  # 2.0 dB/km
                elif sweep_param == "intrinsic_error_ed":
                    default_min = NoiseParameters.INTRINSIC_ERROR_ED_MIN
                    default_max = NoiseParameters.INTRINSIC_ERROR_ED_MAX
                elif sweep_param == "dark_prob":
                    default_min = DetectorParameters.DARK_PROB_MIN
                    default_max = DetectorParameters.DARK_PROB_MAX
                elif sweep_param == "jitter_ns":
                    default_min = DetectorParameters.JITTER_NS_MIN
                    default_max = DetectorParameters.JITTER_NS_MAX
                elif sweep_param == "spdc_brightness_mu":
                    default_min = SourceParameters.SPDC_BRIGHTNESS_MU_MIN
                    default_max = SourceParameters.SPDC_BRIGHTNESS_MU_DEFAULT * 5  # 0.5
                else:
                    default_min, default_max = 0.0, 100.0

                sweep_min = st.number_input("Min Value", value=float(default_min),
                                           help="Minimum value for the sweep range")
                sweep_max = st.number_input("Max Value", value=float(default_max),
                                           help="Maximum value for the sweep range")

                # Validate min < max
                if sweep_min >= sweep_max:
                    st.error("⚠️ Min value must be less than Max value!")

                num_points = st.slider("Number of Points", 5, 50, 20,
                                      help="Number of points to sample between min and max")

                # Generate sweep values (convert to int for num_pairs)
                if sweep_param == "num_pairs":
                    sweep_values = [int(x) for x in np.linspace(sweep_min, sweep_max, num_points)]
                else:
                    sweep_values = np.linspace(sweep_min, sweep_max, num_points).tolist()
            with c2:
                st.markdown("**Preview:**")
                preview_df = pd.DataFrame({"Point": range(1, min(11, len(sweep_values)+1)), "Value": sweep_values[:10]})
                st.dataframe(preview_df, width='stretch')
                st.caption(f"📊 Total points: {len(sweep_values)}")
                est_time = len(sweep_values) * estimate_runtime(base_config.num_pairs, base_config.backend) / 1000
                st.caption(f"⏱️ Estimated time: ~{est_time:.1f}s")

            if st.button("Run Sweep", type="primary", disabled=(sweep_min >= sweep_max),
                        help="Execute a parametric sweep varying the selected parameter across the specified range"):
                pbar = st.progress(0)
                status = st.empty()
                def pcb(p, m):
                    pbar.progress(p)
                    status.text(m)
                try:
                    sweep_results = run_parameter_sweep(base_config, sweep_param, sweep_values, progress_callback=pcb)
                    st.session_state.sweep_results = sweep_results
                    st.session_state.sweep_param = sweep_param_label  # Store the label, not the internal name
                    st.session_state.sweep_history.append({
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'parameter': sweep_param_label,
                        'results': sweep_results
                    })
                    pbar.empty()
                    status.empty()
                    st.success(f"✅ Sweep completed! {len(sweep_results)} points analyzed.")
                    st.rerun()
                except Exception as e:
                    pbar.empty()
                    status.empty()
                    st.error(f"❌ Sweep failed: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
            if 'sweep_results' in st.session_state:
                st.markdown("---")
                sweep_results = st.session_state.sweep_results
                sweep_param = st.session_state.sweep_param
                label = sweep_param.replace('_', ' ').title()
                figs = create_sweep_plots(sweep_results, sweep_param, label)
                # Show all plots simultaneously in a grid
                colA, colB, colC = st.columns(3)
                with colA:
                    st.pyplot(figs['s_vs_qber'], width='stretch')
                with colB:
                    st.pyplot(figs['keyrate_vs_qber'], width='stretch')
                with colC:
                    st.pyplot(figs['keyrate_vs_param'], width='stretch')
                colD, colE = st.columns(2)
                with colD:
                    st.pyplot(figs['chsh_vs_param'], width='stretch')
                with colE:
                    st.pyplot(figs['detection_vs_param'], width='stretch')
                with st.expander("How to read these plots", expanded=False):
                    st.markdown(
                        "- S vs QBER: tradeoff between Bell violation (S) and errors (QBER). Secure region: S>2 and QBER<11%.\n"
                        "- Key Rate vs QBER: asymptotic and finite-key rates vs error.\n"
                        f"- Key Rate vs {label}: sensitivity of secret rate to the swept parameter.\n"
                        f"- CHSH vs {label}: Bell violation vs parameter.\n"
                        f"- Detection vs {label}: throughput (loss + detector effects)."
                    )
                st.markdown("---")
                st.markdown("### Sweep Summary")
                rows = [{label: val, 'CHSH S': r.chsh_S, 'QBER': r.qber, 'Key Rate (Asymp)': r.key_rate_asymptotic, 'Key Rate (Finite)': r.key_rate_finite, 'Detection Eff': r.detection_efficiency} for val, r in sweep_results]
                df_sum = pd.DataFrame(rows)
                st.dataframe(df_sum, width='stretch')
                buf2 = io.StringIO()
                df_sum.to_csv(buf2, index=False)
                st.download_button("Download Summary (CSV)", buf2.getvalue(), "sweep_summary.csv", "text/csv")

    # ========================================================================
    # TAB 2: EXPERIMENT (Main QKD Simulation)
    # ========================================================================
    # Configure and run E91 QKD experiments with full parameter control
    # ========================================================================
    with tab_exp:
        st.subheader("Experiment Configuration & Results")

        col_preset, col_backend, col_seed = st.columns(3)
        with col_preset:
            preset_options = [
                "Custom",
                "Ideal (No Losses)",
                "Low Noise (alpha=0.06)",
                "Moderate Fiber (10km)",
                "Long Distance (50km)",
                "Realistic Lab",
                "Satellite LEO",
            ]
            # Get current preset index
            try:
                current_index = preset_options.index(st.session_state.current_preset)
            except ValueError:
                current_index = 0  # Default to "Custom"

            preset = st.selectbox(
                "Preset",
                preset_options,
                index=current_index,
                help="Select a preset configuration. All parameters will update automatically.",
                key="preset_selector"
            )

            # If preset changed, update all parameters in session state
            if preset != st.session_state.current_preset and preset != "Custom":
                st.session_state.current_preset = preset
                apply_preset_to_session_state(preset)
                st.rerun()
            elif preset == "Custom":
                st.session_state.current_preset = "Custom"
        with col_backend:
            # Get available backends using dependency checker
            backends = get_available_backends(deps)

            if not backends:
                st.error("❌ No quantum backend available!")
                st.info("Please install dependencies using the commands in the 'Backend Dependencies Status' section above.")
                st.stop()

            backend = st.selectbox("Backend", backends, help=h("backend"))

            # Validate backend selection
            is_valid, error_msg = validate_backend_selection(backend, deps)
            if not is_valid:
                st.error(f"❌ {error_msg}")
                st.info("Please install the required dependencies using the commands above.")
                st.stop()
        with col_seed:
            seed_col1, seed_col2 = st.columns([5, 1])
            with seed_col1:
                # Use config constants for seed range
                seed = st.number_input("Random Seed",
                                      min_value=0,
                                      max_value=9999,  # Practical UI limit
                                      value=int(st.session_state["seed"]),
                                      help="Random seed for reproducibility")
                st.session_state["seed"] = int(seed)
            with seed_col2:
                st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
                if st.button("🎲", help="Randomize seed", key="randomize_seed_btn"):
                    # Random seed upper bound (10000 to include 9999 in range)
                    st.session_state["seed"] = int(np.random.randint(0, 10000))
                    st.rerun()

        st.markdown("---")

        col_inputs, col_losses, col_adv, col_prev = st.columns([1, 1, 1, 1])

        with col_inputs:
            with st.expander("INPUTS", expanded=True):
                # Distance controls with individual settings for Alice and Bob
                st.markdown("**Distances (km):**")
                dist_mode = st.radio("Distance Mode", ["Total (Split)", "Individual"], horizontal=True,
                                    help="Total mode splits distance equally; Individual allows separate Alice/Bob distances.")

                if dist_mode == "Total (Split)":
                    # Calculate total distance from session state
                    total_from_session = float(st.session_state.get('distance_A', 0.0)) + float(st.session_state.get('distance_B', 0.0))
                    total_distance = st.number_input("Total Distance (km)", min_value=0.0, value=total_from_session, step=1.0, format="%.2f", help=h("total_distance"))
                    distance_A = distance_B = total_distance / 2.0
                    # Update session state with new values
                    st.session_state['distance_A'] = distance_A
                    st.session_state['distance_B'] = distance_B
                    st.caption(f"📍 Alice: {distance_A:.1f} km | Bob: {distance_B:.1f} km (split equally from source)")
                else:
                    # Individual distance controls with session state
                    col_dist1, col_dist2, col_dist3 = st.columns([1, 1, 0.5])
                    with col_dist1:
                        distance_A = st.number_input("Alice Distance (km)", min_value=0.0, value=float(st.session_state['distance_A']),
                                                     step=1.0, format="%.2f", help="Distance from source to Alice")
                        st.session_state['distance_A'] = distance_A
                    with col_dist2:
                        distance_B = st.number_input("Bob Distance (km)", min_value=0.0, value=float(st.session_state['distance_B']),
                                                     step=1.0, format="%.2f", help="Distance from source to Bob")
                        st.session_state['distance_B'] = distance_B
                    with col_dist3:
                        # Randomize using practical range for fiber QKD (0-50 km)
                        if st.button("🎲", width='stretch', help="Randomize both Alice and Bob distances (0-50 km)"):
                            st.session_state['distance_A'] = float(np.random.uniform(ChannelLossParameters.DISTANCE_KM_A_MIN, 50.0))
                            st.session_state['distance_B'] = float(np.random.uniform(ChannelLossParameters.DISTANCE_KM_B_MIN, 50.0))
                            st.rerun()
                    st.caption(f"📍 Alice: {distance_A:.1f} km | Bob: {distance_B:.1f} km | Total: {distance_A + distance_B:.1f} km")

                st.markdown("**Angles (radians):**")
                angle_type = st.radio("Type", ["Exact", "Whole Radians (π/n)"], horizontal=True, help=h("angle_type"))
                if st.button("Randomize Angles", width='stretch', help=h("randomize_angles")):
                    if angle_type == "Whole Radians (π/n)":
                        # π/8 increments: 0, π/8, 2π/8, ..., 8π/8 = π (9 values total)
                        for key in ["alice_0", "alice_1", "alice_2", "bob_0", "bob_1", "bob_2"]:
                            st.session_state[key] = float(np.random.choice(range(9)) * np.pi / 8)
                    else:
                        # Random angles in [0, π] using validation range
                        for key in ["alice_0", "alice_1", "alice_2", "bob_0", "bob_1", "bob_2"]:
                            st.session_state[key] = float(np.random.uniform(ValidationRanges.ANGLE_MIN, np.pi))
                    st.rerun()
                # Angle editors
                if angle_type == "Whole Radians (π/n)":
                    # Use selectboxes with symbolic π labels at π/8 increments (0 to π)
                    opts = [(f"{k}π/8" if k > 0 else "0", k*np.pi/8.0) for k in range(0, 9)]
                    label_map = {v: l for l, v in opts}
                    a1, a2, a3 = st.columns(3)
                    with a1:
                        alice_0 = st.selectbox("θ_A0", opts, index=int(round(st.session_state.get('alice_0', 0.0)/(np.pi/8))), format_func=lambda x: x[0], help=h("angle"))[1]
                    with a2:
                        alice_1 = st.selectbox("θ_A1", opts, index=int(round(st.session_state.get('alice_1', np.pi/8)/(np.pi/8))), format_func=lambda x: x[0], help=h("angle"))[1]
                    with a3:
                        alice_2 = st.selectbox("θ_A2", opts, index=int(round(st.session_state.get('alice_2', np.pi/4)/(np.pi/8))), format_func=lambda x: x[0], help=h("angle"))[1]
                    b1, b2, b3 = st.columns(3)
                    with b1:
                        bob_0 = st.selectbox("θ_B0", opts, index=int(round(st.session_state.get('bob_0', np.pi/8)/(np.pi/8))), format_func=lambda x: x[0], help=h("angle"))[1]
                    with b2:
                        bob_1 = st.selectbox("θ_B1", opts, index=int(round(st.session_state.get('bob_1', np.pi/4)/(np.pi/8))), format_func=lambda x: x[0], help=h("angle"))[1]
                    with b3:
                        bob_2 = st.selectbox("θ_B2", opts, index=int(round(st.session_state.get('bob_2', 3*np.pi/8)/(np.pi/8))), format_func=lambda x: x[0], help=h("angle"))[1]
                else:
                    # Exact angle input with validation ranges (0 to π for E91)
                    a1, a2, a3 = st.columns(3)
                    with a1:
                        alice_0 = st.number_input("theta_A0", ValidationRanges.ANGLE_MIN, float(np.pi), st.session_state.get("alice_0", ALICE_ANGLES_DEFAULT[0]), 0.05, format="%.4f", help=h("angle"))
                    with a2:
                        alice_1 = st.number_input("theta_A1", ValidationRanges.ANGLE_MIN, float(np.pi), st.session_state.get("alice_1", ALICE_ANGLES_DEFAULT[1]), 0.05, format="%.4f", help=h("angle"))
                    with a3:
                        alice_2 = st.number_input("theta_A2", ValidationRanges.ANGLE_MIN, float(np.pi), st.session_state.get("alice_2", ALICE_ANGLES_DEFAULT[2]), 0.05, format="%.4f", help=h("angle"))
                    b1, b2, b3 = st.columns(3)
                    with b1:
                        bob_0 = st.number_input("theta_B0", ValidationRanges.ANGLE_MIN, float(np.pi), st.session_state.get("bob_0", BOB_ANGLES_DEFAULT[0]), 0.05, format="%.4f", help=h("angle"))
                    with b2:
                        bob_1 = st.number_input("theta_B1", ValidationRanges.ANGLE_MIN, float(np.pi), st.session_state.get("bob_1", BOB_ANGLES_DEFAULT[1]), 0.05, format="%.4f", help=h("angle"))
                    with b3:
                        bob_2 = st.number_input("theta_B2", ValidationRanges.ANGLE_MIN, float(np.pi), st.session_state.get("bob_2", BOB_ANGLES_DEFAULT[2]), 0.05, format="%.4f", help=h("angle"))

                st.session_state.update({'alice_0': alice_0, 'alice_1': alice_1, 'alice_2': alice_2, 'bob_0': bob_0, 'bob_1': bob_1, 'bob_2': bob_2})

                # Enhanced Bell state selection with visualization
                st.markdown("---")
                st.markdown("**🔗 Bell State Selection**")
                bell_labels = {
                    "Phi+ (|00>+|11>)/√2": "phi_plus",
                    "Phi- (|00>-|11>)/√2": "phi_minus",
                    "Psi+ (|01>+|10>)/√2": "psi_plus",
                    "Psi- (|01>-|10>)/√2": "psi_minus",
                }
                bell_descriptions = {
                    "phi_plus": "Maximally entangled state (symmetric). Standard for E91 QKD.",
                    "phi_minus": "Maximally entangled with phase flip. Tests protocol robustness.",
                    "psi_plus": "Maximally entangled (antisymmetric). Alternative encoding.",
                    "psi_minus": "Singlet state. Antisymmetric with phase flip.",
                }

                bell_choice = st.selectbox(
                    "Bell State",
                    list(bell_labels.keys()),
                    index=0,
                    help="Select initial entangled state (Qiskit backend). Phi+: Standard symmetric state for E91. Phi-: Phase-flipped variant. Psi+: Antisymmetric encoding. Psi-: Singlet state.",
                )
                bell_state = bell_labels[bell_choice]
                st.caption(f"💡 {bell_descriptions[bell_state]}")

                # Use config constants for num_pairs range
                num_pairs = st.number_input("Photon Pairs",
                                           min_value=CoreParameters.NUM_PAIRS_MIN,
                                           max_value=min(1000000, CoreParameters.NUM_PAIRS_MAX),  # Practical UI limit
                                           value=st.session_state.get('num_pairs', 10000),
                                           step=1000,
                                           help=h("num_pairs"))
                st.session_state['num_pairs'] = int(num_pairs)
                est_time = estimate_runtime(num_pairs, backend)

                st.markdown("---")
                st.markdown("### Basis Probabilities")
                # Default basis probabilities for E91 (uniform over first two bases)
                DEFAULT_BASIS_PROBS = [0.5, 0.5, 0.0]
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Randomize A & B", width='stretch', help=h("prob_randomize")):
                        st.session_state['pA'] = normalize_probs(np.random.rand(3).tolist())
                        st.session_state['pB'] = normalize_probs(np.random.rand(3).tolist())
                with c2:
                    if st.button("Reset Defaults", width='stretch', help=h("prob_reset")):
                        st.session_state['pA'] = DEFAULT_BASIS_PROBS
                        st.session_state['pB'] = DEFAULT_BASIS_PROBS
                pA = st.session_state.get('pA', DEFAULT_BASIS_PROBS)
                pB = st.session_state.get('pB', DEFAULT_BASIS_PROBS)
                la, lb = st.columns(2)
                with la:
                    # Use validation ranges for probabilities
                    pA0 = st.slider("p(A0)", ValidationRanges.PROBABILITY_MIN, ValidationRanges.PROBABILITY_MAX, float(pA[0]), 0.01, help=h("pA"))
                    pA1 = st.slider("p(A1)", ValidationRanges.PROBABILITY_MIN, ValidationRanges.PROBABILITY_MAX, float(pA[1]), 0.01, help=h("pA"))
                    pA2 = st.slider("p(A2)", ValidationRanges.PROBABILITY_MIN, ValidationRanges.PROBABILITY_MAX, float(pA[2]), 0.01, help=h("pA"))
                with lb:
                    pB0 = st.slider("p(B0)", ValidationRanges.PROBABILITY_MIN, ValidationRanges.PROBABILITY_MAX, float(pB[0]), 0.01, help=h("pB"))
                    pB1 = st.slider("p(B1)", ValidationRanges.PROBABILITY_MIN, ValidationRanges.PROBABILITY_MAX, float(pB[1]), 0.01, help=h("pB"))
                    pB2 = st.slider("p(B2)", ValidationRanges.PROBABILITY_MIN, ValidationRanges.PROBABILITY_MAX, float(pB[2]), 0.01, help=h("pB"))
                pA = normalize_probs([pA0, pA1, pA2])
                pB = normalize_probs([pB0, pB1, pB2])
                st.caption(f"Alice p = {np.round(pA, 3).tolist()} | Bob p = {np.round(pB, 3).tolist()}")

        with col_losses:
            with st.expander("LOSSES", expanded=True):
                st.markdown("**Logical Noise**")
                enable_depol = st.checkbox("Depolarizing Noise", value=st.session_state.enable_depol, help=h("enable_depol"), on_change=switch_to_custom)
                st.session_state.enable_depol = enable_depol
                if enable_depol:
                    # Use config constants for depolarizing noise range
                    depol_alpha = st.slider("alpha",
                                          NoiseParameters.DEPOLARIZING_ALPHA_MIN,
                                          0.3,  # Practical upper limit for QKD
                                          float(st.session_state.depol_alpha),
                                          0.01,
                                          help=h("depol_alpha"))
                    st.session_state.depol_alpha = depol_alpha
                else:
                    depol_alpha = st.session_state.get('depol_alpha', NoiseParameters.DEPOLARIZING_ALPHA_DEFAULT)

                enable_vis = st.checkbox("Visibility Reduction", value=st.session_state.enable_vis, help=h("enable_visibility"), on_change=switch_to_custom)
                st.session_state.enable_vis = enable_vis
                if enable_vis:
                    # Use config constants for visibility range
                    visibility = st.slider("V",
                                         0.5,  # Practical lower bound for QKD
                                         NoiseParameters.VISIBILITY_MAX,
                                         float(st.session_state.visibility),
                                         0.01,
                                         help=h("visibility"))
                    st.session_state.visibility = visibility
                else:
                    visibility = st.session_state.get('visibility', NoiseParameters.VISIBILITY_DEFAULT)

                enable_intrinsic = st.checkbox("Intrinsic Errors", value=st.session_state.enable_intrinsic, help=h("enable_intrinsic"), on_change=switch_to_custom)
                st.session_state.enable_intrinsic = enable_intrinsic
                if enable_intrinsic:
                    # Use config constants for intrinsic error range
                    intrinsic_ed = st.slider("e_d",
                                            NoiseParameters.INTRINSIC_ERROR_ED_MIN,
                                            0.1,  # Practical upper limit
                                            float(st.session_state.intrinsic_ed),
                                            0.001,
                                            help=h("intrinsic_ed"))
                    st.session_state.intrinsic_ed = intrinsic_ed
                else:
                    intrinsic_ed = st.session_state.get('intrinsic_ed', NoiseParameters.INTRINSIC_ERROR_ED_DEFAULT)

                st.markdown("**Channel Loss**")
                enable_fiber = st.checkbox("Fiber Loss", value=st.session_state.enable_fiber, help=h("enable_fiber"), on_change=switch_to_custom)
                st.session_state.enable_fiber = enable_fiber
                if enable_fiber:
                    # Use config constants for fiber loss range
                    fiber_loss_db = st.slider("Fiber Loss (dB/km)",
                                             0.1,  # Practical lower limit for fiber
                                             ChannelLossParameters.FIBER_LOSS_DB_PER_KM_MAX,
                                             float(st.session_state.get('fiber_loss_db', ChannelLossParameters.FIBER_LOSS_DB_PER_KM_DEFAULT)),
                                             0.1,
                                             help=h("fiber_loss_db"))
                    st.session_state['fiber_loss_db'] = fiber_loss_db
                    colfa, colfb = st.columns(2)
                    with colfa:
                        # Use config constants for fixed loss range
                        loss_a = st.number_input("Fixed Loss A (dB)",
                                                ChannelLossParameters.LOSS_DB_A_MIN,
                                                ChannelLossParameters.LOSS_DB_A_MAX,
                                                float(st.session_state.get('loss_a', ChannelLossParameters.LOSS_DB_A_DEFAULT)),
                                                0.1,
                                                help=h("fixed_loss_a"))
                        st.session_state['loss_a'] = loss_a
                    with colfb:
                        loss_b = st.number_input("Fixed Loss B (dB)",
                                                ChannelLossParameters.LOSS_DB_B_MIN,
                                                ChannelLossParameters.LOSS_DB_B_MAX,
                                                float(st.session_state.get('loss_b', ChannelLossParameters.LOSS_DB_B_DEFAULT)),
                                                0.1,
                                                help=h("fixed_loss_b"))
                        st.session_state['loss_b'] = loss_b
                else:
                    fiber_loss_db = st.session_state.get('fiber_loss_db', ChannelLossParameters.FIBER_LOSS_DB_PER_KM_DEFAULT)
                    loss_a = st.session_state.get('loss_a', ChannelLossParameters.LOSS_DB_A_DEFAULT)
                    loss_b = st.session_state.get('loss_b', ChannelLossParameters.LOSS_DB_B_DEFAULT)

                # Insertion Loss controls (new addition - previously only in presets)
                enable_insertion = st.checkbox("Insertion Loss",
                                              value=st.session_state.get('enable_insertion', ChannelLossParameters.ENABLE_INSERTION_LOSS_DEFAULT),
                                              help=h("enable_insertion"),
                                              on_change=switch_to_custom)
                st.session_state.enable_insertion = enable_insertion
                if enable_insertion:
                    insertion_loss = st.slider("Insertion Loss (dB)",
                                              ChannelLossParameters.INSERTION_LOSS_DB_MIN,
                                              ChannelLossParameters.INSERTION_LOSS_DB_MAX,
                                              float(st.session_state.get('insertion_loss', ChannelLossParameters.INSERTION_LOSS_DB_DEFAULT)),
                                              0.1,
                                              help=h("insertion_loss"))
                    st.session_state['insertion_loss'] = insertion_loss
                else:
                    insertion_loss = st.session_state.get('insertion_loss', ChannelLossParameters.INSERTION_LOSS_DB_DEFAULT)

                # Polarization Misalignment controls (new addition - previously only in presets)
                enable_misalignment = st.checkbox("Polarization Misalignment",
                                                 value=st.session_state.get('enable_misalignment', ChannelLossParameters.ENABLE_MISALIGNMENT_DEFAULT),
                                                 help=h("enable_misalignment"),
                                                 on_change=switch_to_custom)
                st.session_state.enable_misalignment = enable_misalignment
                if enable_misalignment:
                    pol_drift = st.slider("Polarization Drift (degrees)",
                                         ChannelLossParameters.POLARIZATION_DRIFT_DEG_MIN,
                                         ChannelLossParameters.POLARIZATION_DRIFT_DEG_MAX,
                                         float(st.session_state.get('pol_drift', ChannelLossParameters.POLARIZATION_DRIFT_DEG_DEFAULT)),
                                         0.5,
                                         help=h("pol_drift"))
                    st.session_state['pol_drift'] = pol_drift
                else:
                    pol_drift = st.session_state.get('pol_drift', ChannelLossParameters.POLARIZATION_DRIFT_DEG_DEFAULT)

                # Enhanced Quantum Repeater controls
                st.markdown("---")
                st.markdown("**🔄 Quantum Repeaters**")
                enable_repeater = st.checkbox("Enable Quantum Repeaters", value=st.session_state.enable_repeater, on_change=switch_to_custom,
                                              help="Enable quantum repeaters to boost signal and enhance qubit transfer integrity over long distances. Repeaters provide effective gain to compensate fiber loss.")
                st.session_state.enable_repeater = enable_repeater
                if enable_repeater:
                    rep_col1, rep_col2 = st.columns(2)
                    with rep_col1:
                        # Use config constants for repeater range
                        num_repeaters = st.number_input("Number of Repeaters",
                                                       min_value=ChannelLossParameters.NUM_REPEATERS_MIN,
                                                       max_value=20,  # Practical UI limit
                                                       value=int(st.session_state.get('num_repeaters', ChannelLossParameters.NUM_REPEATERS_DEFAULT)),
                                                       step=1,
                                                       help="Number of quantum repeaters deployed along the channel. More repeaters = better signal preservation.")
                        st.session_state['num_repeaters'] = num_repeaters
                    with rep_col2:
                        # Use config constants for repeater gain range
                        repeater_gain_dB = st.number_input("Gain per Repeater (dB)",
                                                           min_value=ChannelLossParameters.REPEATER_GAIN_DB_MIN,
                                                           max_value=10.0,  # Practical limit
                                                           value=float(st.session_state.get('repeater_gain_dB', ChannelLossParameters.REPEATER_GAIN_DB_DEFAULT)),
                                                           step=0.5,
                                                           format="%.1f",
                                                           help="Signal gain provided by each repeater (dB). Typical values: 2-5 dB.")
                        st.session_state['repeater_gain_dB'] = repeater_gain_dB

                    # Show repeater effectiveness
                    total_fiber_loss = (distance_A + distance_B) * fiber_loss_db
                    total_repeater_gain = num_repeaters * repeater_gain_dB
                    net_loss = max(0.0, total_fiber_loss - total_repeater_gain)

                    st.caption(f"📊 Fiber Loss: {total_fiber_loss:.1f} dB | Repeater Gain: {total_repeater_gain:.1f} dB | Net Loss: {net_loss:.1f} dB")

                    # Visual indicator
                    if total_repeater_gain >= total_fiber_loss:
                        st.success("✅ Repeaters fully compensate fiber loss!")
                    elif total_repeater_gain >= total_fiber_loss * 0.5:
                        st.info("🔶 Repeaters partially compensate fiber loss")
                    else:
                        st.warning("⚠️ Fiber loss exceeds repeater gain")
                else:
                    num_repeaters = st.session_state.get('num_repeaters', 0)
                    repeater_gain_dB = st.session_state.get('repeater_gain_dB', 0.0)
                    enable_repeater = False

                st.markdown("**Detector Effects**")
                enable_detector = st.checkbox("Detector Inefficiency", value=st.session_state.enable_detector, help=h("enable_detector"), on_change=switch_to_custom)
                st.session_state.enable_detector = enable_detector
                if enable_detector:
                    # Use config constants for detector efficiency ranges
                    herald_eff = st.slider("Heralding eta",
                                         0.1,  # Practical lower limit
                                         DetectorParameters.HERALDING_EFFICIENCY_MAX,
                                         float(st.session_state.herald_eff),
                                         0.05,
                                         help=h("herald_eff"))
                    st.session_state.herald_eff = herald_eff
                    end_det_eff = st.slider("End Detector eta",
                                           0.1,  # Practical lower limit
                                           DetectorParameters.END_DETECTOR_EFFICIENCY_MAX,
                                           float(st.session_state.end_det_eff),
                                           0.05,
                                           help=h("end_det_eff"))
                    st.session_state.end_det_eff = end_det_eff
                else:
                    herald_eff = st.session_state.get('herald_eff', DetectorParameters.HERALDING_EFFICIENCY_DEFAULT)
                    end_det_eff = st.session_state.get('end_det_eff', DetectorParameters.END_DETECTOR_EFFICIENCY_DEFAULT)

                enable_dark = st.checkbox("Dark Counts", value=st.session_state.enable_dark, help=h("enable_dark"), on_change=switch_to_custom)
                st.session_state.enable_dark = enable_dark
                if enable_dark:
                    use_cps = st.checkbox("Use Rate (CPS)", value=st.session_state.get('use_cps', DetectorParameters.USE_DARK_CPS_DEFAULT), help=h("use_cps"))
                    st.session_state['use_cps'] = use_cps
                    if use_cps:
                        # Use config constants for dark CPS range
                        dark_cps = st.number_input("Dark CPS",
                                                  DetectorParameters.DARK_CPS_MIN,
                                                  DetectorParameters.DARK_CPS_MAX,
                                                  float(st.session_state.get('dark_cps', DetectorParameters.DARK_CPS_DEFAULT)),
                                                  1.0,
                                                  format="%.2f",
                                                  help=h("dark_cps"))
                        st.session_state['dark_cps'] = dark_cps
                        dark_prob = st.session_state.get('dark_prob', DetectorParameters.DARK_PROB_DEFAULT)
                    else:
                        # Use config constants for dark probability range
                        dark_prob = st.number_input("Dark Probability",
                                                   DetectorParameters.DARK_PROB_MIN,
                                                   DetectorParameters.DARK_PROB_MAX,
                                                   float(st.session_state.get('dark_prob', DetectorParameters.DARK_PROB_DEFAULT)),
                                                   0.0001,
                                                   format="%.4f",
                                                   help=h("dark_prob"))
                        st.session_state['dark_prob'] = dark_prob
                        dark_cps = st.session_state.get('dark_cps', DetectorParameters.DARK_CPS_DEFAULT)
                else:
                    use_cps = st.session_state.get('use_cps', DetectorParameters.USE_DARK_CPS_DEFAULT)
                    dark_cps = st.session_state.get('dark_cps', DetectorParameters.DARK_CPS_DEFAULT)
                    dark_prob = st.session_state.get('dark_prob', DetectorParameters.DARK_PROB_DEFAULT)

                enable_jitter = st.checkbox("Timing Jitter", value=st.session_state.enable_jitter, help=h("enable_jitter"), on_change=switch_to_custom)
                st.session_state.enable_jitter = enable_jitter
                if enable_jitter:
                    # Use config constants for jitter range
                    jitter_ns = st.slider("Jitter (ns)",
                                        0.1,  # Practical lower limit
                                        DetectorParameters.JITTER_NS_MAX,
                                        float(st.session_state.get('jitter_ns', DetectorParameters.JITTER_NS_DEFAULT)),
                                        0.1,
                                        help=h("jitter_ns"))
                    st.session_state['jitter_ns'] = jitter_ns
                else:
                    jitter_ns = st.session_state.get('jitter_ns', DetectorParameters.JITTER_NS_DEFAULT)

                enable_deadtime = st.checkbox("Detector Deadtime", value=st.session_state.enable_deadtime, help=h("enable_deadtime"), on_change=switch_to_custom)
                st.session_state.enable_deadtime = enable_deadtime
                if enable_deadtime:
                    # Use config constants for deadtime range
                    deadtime_ns = st.slider("Deadtime (ns)",
                                          DetectorParameters.DEADTIME_NS_MIN + 10.0,  # Practical minimum
                                          DetectorParameters.DEADTIME_NS_MAX,
                                          float(st.session_state.get('deadtime_ns', DetectorParameters.DEADTIME_NS_DEFAULT)),
                                          1.0,
                                          help=h("deadtime_ns"))
                    st.session_state['deadtime_ns'] = deadtime_ns
                else:
                    deadtime_ns = st.session_state.get('deadtime_ns', DetectorParameters.DEADTIME_NS_DEFAULT)

                enable_after = st.checkbox("Afterpulsing", value=st.session_state.enable_after, help=h("enable_after"), on_change=switch_to_custom)
                st.session_state.enable_after = enable_after
                if enable_after:
                    # Use config constants for afterpulsing range
                    after_prob = st.slider("Afterpulse Probability",
                                         DetectorParameters.AFTERPULSING_PROB_MIN,
                                         DetectorParameters.AFTERPULSING_PROB_MAX,
                                         float(st.session_state.get('after_prob', DetectorParameters.AFTERPULSING_PROB_DEFAULT)),
                                         0.005,
                                         help=h("after_prob"))
                    st.session_state['after_prob'] = after_prob
                    after_delay = st.slider("Afterpulse Delay (ns)",
                                          DetectorParameters.AFTERPULSING_DELAY_NS_MIN,
                                          DetectorParameters.AFTERPULSING_DELAY_NS_MAX,
                                          float(st.session_state.get('after_delay', DetectorParameters.AFTERPULSING_DELAY_NS_DEFAULT)),
                                          5.0,
                                          help=h("after_delay"))
                    st.session_state['after_delay'] = after_delay
                else:
                    after_prob = st.session_state.get('after_prob', DetectorParameters.AFTERPULSING_PROB_DEFAULT)
                    after_delay = st.session_state.get('after_delay', DetectorParameters.AFTERPULSING_DELAY_NS_DEFAULT)

                enable_sat = st.checkbox("Saturation", value=st.session_state.enable_sat, help=h("enable_sat"), on_change=switch_to_custom)
                st.session_state.enable_sat = enable_sat
                if enable_sat:
                    # Use config constants for saturation rate range
                    saturation_rate = st.number_input("Saturation Rate (Hz)",
                                                     DetectorParameters.SATURATION_RATE_MIN,
                                                     DetectorParameters.SATURATION_RATE_MAX,
                                                     float(st.session_state.get('saturation_rate', DetectorParameters.SATURATION_RATE_DEFAULT)),
                                                     step=1e5,
                                                     format="%.0f",
                                                     help=h("saturation_rate"))
                    st.session_state['saturation_rate'] = saturation_rate
                else:
                    saturation_rate = st.session_state.get('saturation_rate', DetectorParameters.SATURATION_RATE_DEFAULT)

                st.markdown("**Background & Accidentals**")
                enable_bg = st.checkbox("Background Noise", value=st.session_state.enable_bg, help=h("enable_bg"), on_change=switch_to_custom)
                st.session_state.enable_bg = enable_bg
                if enable_bg:
                    # Use config constants for background parameters
                    Y0 = st.number_input("Y0",
                                        BackgroundParameters.Y0_MIN,
                                        BackgroundParameters.Y0_MAX,
                                        float(st.session_state.get('Y0', BackgroundParameters.Y0_DEFAULT)),
                                        format="%.2e",
                                        help=h("Y0"))
                    st.session_state['Y0'] = Y0
                    bg_cps = st.number_input("Background CPS",
                                            BackgroundParameters.BACKGROUND_CPS_MIN,
                                            BackgroundParameters.BACKGROUND_CPS_MAX,
                                            float(st.session_state.get('bg_cps', BackgroundParameters.BACKGROUND_CPS_DEFAULT)),
                                            10.0,
                                            format="%.2f",
                                            help=h("bg_cps"))
                    st.session_state['bg_cps'] = bg_cps
                    coinc_ns = st.slider("Coincidence Window (ns)",
                                       BackgroundParameters.COINCIDENCE_WINDOW_NS_MIN,
                                       BackgroundParameters.COINCIDENCE_WINDOW_NS_MAX,
                                       float(st.session_state.get('coinc_ns', BackgroundParameters.COINCIDENCE_WINDOW_NS_DEFAULT)),
                                       0.1,
                                       help=h("coinc_ns"))
                    st.session_state['coinc_ns'] = coinc_ns
                    rep_rate = st.number_input("Repetition Rate (Hz)",
                                              BackgroundParameters.REPETITION_RATE_HZ_MIN,
                                              BackgroundParameters.REPETITION_RATE_HZ_MAX,
                                              float(st.session_state.get('rep_rate', BackgroundParameters.REPETITION_RATE_HZ_DEFAULT)),
                                              step=1e5,
                                              format="%.0f",
                                              help=h("rep_rate"))
                    st.session_state['rep_rate'] = rep_rate
                    enable_timetag = st.checkbox("Enable Time-Tagging",
                                                value=st.session_state.get('enable_timetag', BackgroundParameters.ENABLE_TIME_TAGGING_DEFAULT),
                                                help=h("enable_timetag"),
                                                on_change=switch_to_custom)
                    st.session_state['enable_timetag'] = enable_timetag
                else:
                    Y0 = st.session_state.get('Y0', BackgroundParameters.Y0_DEFAULT)
                    bg_cps = st.session_state.get('bg_cps', BackgroundParameters.BACKGROUND_CPS_DEFAULT)
                    coinc_ns = st.session_state.get('coinc_ns', BackgroundParameters.COINCIDENCE_WINDOW_NS_DEFAULT)
                    rep_rate = st.session_state.get('rep_rate', BackgroundParameters.REPETITION_RATE_HZ_DEFAULT)
                    enable_timetag = st.session_state.get('enable_timetag', BackgroundParameters.ENABLE_TIME_TAGGING_DEFAULT)

                st.markdown("**Source Effects**")
                # Multi-pair SPDC (simplified squashing model)
                st.info("ℹ️ Multi-Pair SPDC: Simplified threshold-detector squashing model (Beaudry et al. 2008)")
                enable_multi = st.checkbox(
                    "Enable Multi-Pair SPDC",
                    value=st.session_state.get('enable_multi', False),
                    help="Model multi-pair emissions (k≥2) via double-click squashing. Degrades visibility."
                )
                st.session_state.enable_multi = enable_multi
                if enable_multi:
                    # Use config constants for source parameters
                    spdc_mu = st.slider("SPDC Brightness (mu)",
                                      SourceParameters.SPDC_BRIGHTNESS_MU_MIN,
                                      SourceParameters.SPDC_BRIGHTNESS_MU_MAX,
                                      float(st.session_state.get('spdc_mu', SourceParameters.SPDC_BRIGHTNESS_MU_DEFAULT)),
                                      0.01,
                                      help=h("spdc_mu"))
                    st.session_state['spdc_mu'] = spdc_mu

                    # SPDC distribution choice (thermal vs Poisson)
                    spdc_dist = st.selectbox(
                        "SPDC Distribution",
                        ["thermal", "poisson"],
                        index=0 if st.session_state.get('spdc_dist', 'thermal') == 'thermal' else 1,
                        help="Thermal: single-mode SPDC (narrow filter). Poisson: multi-mode (broadband)."
                    )
                    st.session_state['spdc_dist'] = spdc_dist

                    pair_rate = st.number_input("Pair Rate (Hz)",
                                               SourceParameters.PAIR_RATE_MIN,
                                               SourceParameters.PAIR_RATE_MAX,
                                               float(st.session_state.get('pair_rate', SourceParameters.PAIR_RATE_DEFAULT)),
                                               step=1e5,
                                               format="%.0f",
                                               help=h("pair_rate"))
                    st.session_state['pair_rate'] = pair_rate
                    pump_mw = st.slider("Pump Power (mW)",
                                      SourceParameters.PUMP_POWER_MW_MIN,
                                      SourceParameters.PUMP_POWER_MW_MAX,
                                      float(st.session_state.get('pump_mw', SourceParameters.PUMP_POWER_MW_DEFAULT)),
                                      0.5,
                                      help=h("pump_mw"))
                    st.session_state['pump_mw'] = pump_mw
                    wavelength = st.number_input("Wavelength (nm)",
                                                SourceParameters.WAVELENGTH_NM_MIN,
                                                SourceParameters.WAVELENGTH_NM_MAX,
                                                float(st.session_state.get('wavelength', SourceParameters.WAVELENGTH_NM_DEFAULT)),
                                                1.0,
                                                format="%.2f",
                                                help=h("wavelength"))
                    st.session_state['wavelength'] = wavelength
                    filt_band = st.slider("Filter Bandwidth (nm)",
                                        SourceParameters.FILTER_BANDWIDTH_NM_MIN,
                                        SourceParameters.FILTER_BANDWIDTH_NM_MAX,
                                        float(st.session_state.get('filt_band', SourceParameters.FILTER_BANDWIDTH_NM_DEFAULT)),
                                        0.1,
                                        help=h("filt_band"))
                    st.session_state['filt_band'] = filt_band
                else:
                    spdc_mu = st.session_state.get('spdc_mu', SourceParameters.SPDC_BRIGHTNESS_MU_DEFAULT)
                    spdc_dist = st.session_state.get('spdc_dist', 'thermal')
                    pair_rate = st.session_state.get('pair_rate', SourceParameters.PAIR_RATE_DEFAULT)
                    pump_mw = st.session_state.get('pump_mw', SourceParameters.PUMP_POWER_MW_DEFAULT)
                    wavelength = st.session_state.get('wavelength', SourceParameters.WAVELENGTH_NM_DEFAULT)
                    filt_band = st.session_state.get('filt_band', SourceParameters.FILTER_BANDWIDTH_NM_DEFAULT)

                st.markdown("**Satellite / Free-Space**")
                enable_satellite = st.checkbox("Enable Satellite Model", value=st.session_state.enable_satellite, help=h("enable_satellite"), on_change=switch_to_custom)
                st.session_state.enable_satellite = enable_satellite
                if enable_satellite:
                    # Use config constants for satellite parameters
                    beam_div = st.slider("Beam Divergence (urad)",
                                       SatelliteParameters.BEAM_DIVERGENCE_URAD_MIN,
                                       SatelliteParameters.BEAM_DIVERGENCE_URAD_MAX,
                                       float(st.session_state.get('beam_div', SatelliteParameters.BEAM_DIVERGENCE_URAD_DEFAULT)),
                                       1.0,
                                       help=h("beam_div"))
                    st.session_state['beam_div'] = beam_div
                    pointing = st.slider("Pointing Jitter (urad)",
                                       SatelliteParameters.POINTING_JITTER_URAD_MIN,
                                       SatelliteParameters.POINTING_JITTER_URAD_MAX,
                                       float(st.session_state.get('pointing', SatelliteParameters.POINTING_JITTER_URAD_DEFAULT)),
                                       0.5,
                                       help=h("pointing"))
                    st.session_state['pointing'] = pointing
                    rx_fov = st.slider("Receiver FOV (urad)",
                                     SatelliteParameters.RECEIVER_FOV_URAD_MIN,
                                     SatelliteParameters.RECEIVER_FOV_URAD_MAX,
                                     float(st.session_state.get('rx_fov', SatelliteParameters.RECEIVER_FOV_URAD_DEFAULT)),
                                     1.0,
                                     help=h("rx_fov"))
                    st.session_state['rx_fov'] = rx_fov
                    slant_km = st.slider("Slant Range (km)",
                                       SatelliteParameters.SLANT_RANGE_KM_MIN,
                                       SatelliteParameters.SLANT_RANGE_KM_MAX,
                                       float(st.session_state.get('slant_km', SatelliteParameters.SLANT_RANGE_KM_DEFAULT)),
                                       10.0,
                                       help=h("slant_km"))
                    st.session_state['slant_km'] = slant_km
                    tx_ap = st.slider("TX Aperture (m)",
                                    SatelliteParameters.TRANSMITTER_APERTURE_M_MIN,
                                    SatelliteParameters.TRANSMITTER_APERTURE_M_MAX,
                                    float(st.session_state.get('tx_ap', SatelliteParameters.TRANSMITTER_APERTURE_M_DEFAULT)),
                                    0.01,
                                    help=h("tx_ap"))
                    st.session_state['tx_ap'] = tx_ap
                    rx_ap = st.slider("RX Aperture (m)",
                                    SatelliteParameters.RECEIVER_APERTURE_M_MIN,
                                    SatelliteParameters.RECEIVER_APERTURE_M_MAX,
                                    float(st.session_state.get('rx_ap', SatelliteParameters.RECEIVER_APERTURE_M_DEFAULT)),
                                    0.01,
                                    help=h("rx_ap"))
                    st.session_state['rx_ap'] = rx_ap
                    is_day = st.checkbox("Daytime Operation",
                                       value=st.session_state.get('is_day', SatelliteParameters.IS_DAYTIME_DEFAULT),
                                       help=h("is_day"))
                    st.session_state['is_day'] = is_day
                    if is_day:
                        sat_bg_cps = st.number_input("Satellite BG CPS",
                                                    SatelliteParameters.SATELLITE_BACKGROUND_CPS_MIN,
                                                    SatelliteParameters.SATELLITE_BACKGROUND_CPS_MAX,
                                                    float(st.session_state.get('sat_bg_cps', SatelliteParameters.SATELLITE_BACKGROUND_CPS_DEFAULT)),
                                                    10.0,
                                                    format="%.2f",
                                                    help=h("sat_bg_cps"))
                        st.session_state['sat_bg_cps'] = sat_bg_cps
                    else:
                        sat_bg_cps = st.session_state.get('sat_bg_cps', SatelliteParameters.SATELLITE_BACKGROUND_CPS_DEFAULT)
                    enable_turb = st.checkbox("Atmospheric Turbulence",
                                             value=st.session_state.get('enable_turb', SatelliteParameters.ENABLE_TURBULENCE_DEFAULT),
                                             help=h("enable_turb"),
                                             on_change=switch_to_custom)
                    st.session_state['enable_turb'] = enable_turb
                    if enable_turb:
                        cn2 = st.number_input("Cn2 (turbulence)",
                                            SatelliteParameters.CN2_MIN,
                                            SatelliteParameters.CN2_MAX,
                                            float(st.session_state.get('cn2', SatelliteParameters.CN2_DEFAULT)),
                                            format="%.2e",
                                            help=h("cn2"))
                        st.session_state['cn2'] = cn2
                    else:
                        cn2 = st.session_state.get('cn2', SatelliteParameters.CN2_DEFAULT)
                else:
                    beam_div = st.session_state.get('beam_div', SatelliteParameters.BEAM_DIVERGENCE_URAD_DEFAULT)
                    pointing = st.session_state.get('pointing', SatelliteParameters.POINTING_JITTER_URAD_DEFAULT)
                    rx_fov = st.session_state.get('rx_fov', SatelliteParameters.RECEIVER_FOV_URAD_DEFAULT)
                    slant_km = st.session_state.get('slant_km', SatelliteParameters.SLANT_RANGE_KM_DEFAULT)
                    tx_ap = st.session_state.get('tx_ap', SatelliteParameters.TRANSMITTER_APERTURE_M_DEFAULT)
                    rx_ap = st.session_state.get('rx_ap', SatelliteParameters.RECEIVER_APERTURE_M_DEFAULT)
                    is_day = st.session_state.get('is_day', SatelliteParameters.IS_DAYTIME_DEFAULT)
                    sat_bg_cps = st.session_state.get('sat_bg_cps', SatelliteParameters.SATELLITE_BACKGROUND_CPS_DEFAULT)
                    enable_turb = st.session_state.get('enable_turb', SatelliteParameters.ENABLE_TURBULENCE_DEFAULT)
                    cn2 = st.session_state.get('cn2', SatelliteParameters.CN2_DEFAULT)

        with col_adv:
            with st.expander("ADVANCED", expanded=(st.session_state.ui_mode == "Advanced")):
                st.markdown("**Security Parameters**")
                # Use config constants for security parameters
                epsilon_sec = st.number_input("epsilon_sec",
                                             SecurityParameters.EPSILON_SEC_MIN,
                                             SecurityParameters.EPSILON_SEC_MAX,
                                             float(st.session_state.get('epsilon_sec', SecurityParameters.EPSILON_SEC_DEFAULT)),
                                             format="%.2e",
                                             help=h("epsilon_sec"))
                st.session_state['epsilon_sec'] = epsilon_sec
                epsilon_cor = st.number_input("epsilon_cor",
                                             SecurityParameters.EPSILON_COR_MIN,
                                             SecurityParameters.EPSILON_COR_MAX,
                                             float(st.session_state.get('epsilon_cor', SecurityParameters.EPSILON_COR_DEFAULT)),
                                             format="%.2e",
                                             help=h("epsilon_cor"))
                st.session_state['epsilon_cor'] = epsilon_cor
                pe_fraction = st.slider("PE Fraction",
                                       SecurityParameters.PE_FRACTION_MIN,
                                       SecurityParameters.PE_FRACTION_MAX,
                                       float(st.session_state.get('pe_fraction', SecurityParameters.PE_FRACTION_DEFAULT)),
                                       0.05,
                                       help=h("pe_fraction"))
                st.session_state['pe_fraction'] = pe_fraction
                st.markdown("**Information Reconciliation**")
                ir_protocol = st.selectbox("IR Protocol",
                                          ["BBBSS", "Cascade"],
                                          index=0 if st.session_state.get('ir_protocol', SecurityParameters.IR_DEFAULT) == 'BBBSS' else 1,
                                          help=h("ir_protocol"))
                st.session_state['ir_protocol'] = ir_protocol
                # Use config constants for EC parameters
                f_EC = st.slider("f_EC",
                               SecurityParameters.F_EC_MIN,
                               SecurityParameters.F_EC_MAX,
                               float(st.session_state.get('f_EC', SecurityParameters.F_EC_DEFAULT)),
                               0.05,
                               help=h("f_EC"))
                st.session_state['f_EC'] = f_EC
                block_length = st.number_input("Block Length",
                                              SecurityParameters.BLOCK_LENGTH_MIN,
                                              SecurityParameters.BLOCK_LENGTH_MAX,
                                              int(st.session_state.get('block_length', SecurityParameters.BLOCK_LENGTH_DEFAULT)),
                                              64,
                                              help=h("block_length"))
                st.session_state['block_length'] = block_length
                code_rate = st.slider("Code Rate (PA)",
                                    SecurityParameters.CODE_RATE_MIN,
                                    SecurityParameters.CODE_RATE_MAX,
                                    float(st.session_state.get('code_rate', SecurityParameters.CODE_RATE_DEFAULT)),
                                    0.01,
                                    help=h("code_rate"))
                st.session_state['code_rate'] = code_rate
                st.markdown("**Device-Independent**")
                enable_routed_di = st.checkbox("Enable Routed-DI",
                                              value=st.session_state.get('enable_routed_di', DeviceIndependentParameters.ENABLE_ROUTED_DI_DEFAULT),
                                              help=h("enable_routed_di"),
                                              on_change=switch_to_custom)
                st.session_state['enable_routed_di'] = enable_routed_di
                di_bound_type = st.selectbox("DI Bound Type",
                                            ["default_di_bound"],
                                            index=0,
                                            help=h("di_bound_type"))
                st.session_state['di_bound_type'] = di_bound_type
                # Use config constants for DI routing efficiencies
                routing_f = st.slider("eta_f (Fixed Routing)",
                                    DeviceIndependentParameters.ROUTING_EFFICIENCY_F_MIN,
                                    DeviceIndependentParameters.ROUTING_EFFICIENCY_F_MAX,
                                    float(st.session_state.get('routing_f', DeviceIndependentParameters.ROUTING_EFFICIENCY_F_DEFAULT)),
                                    0.01,
                                    help=h("routing_f"))
                st.session_state['routing_f'] = routing_f
                routing_d = st.slider("eta_d (Distance-Dependent)",
                                    DeviceIndependentParameters.ROUTING_EFFICIENCY_D_MIN,
                                    DeviceIndependentParameters.ROUTING_EFFICIENCY_D_MAX,
                                    float(st.session_state.get('routing_d', DeviceIndependentParameters.ROUTING_EFFICIENCY_D_DEFAULT)),
                                    0.01,
                                    help=h("routing_d"))
                st.session_state['routing_d'] = routing_d

                st.markdown("**Adversary (Eve)**")
                enable_eve = st.checkbox("Enable Eavesdropper (Eve)",
                                        value=st.session_state.enable_eve,
                                        help=h("enable_eve"),
                                        on_change=switch_to_custom)
                st.session_state.enable_eve = enable_eve
                if enable_eve:
                    eve_model = st.selectbox("Eve Model",
                                           ["intercept_resend"],
                                           index=0 if st.session_state.get('eve_model', EavesdropperParameters.EVE_MODEL_DEFAULT) == 'intercept_resend' else 0,
                                           help=h("eve_model"))
                    st.session_state['eve_model'] = eve_model
                    # Use config constants for Eve intercept probability
                    eve_prob = st.slider("Intercept Probability",
                                       EavesdropperParameters.EVE_INTERCEPT_PROB_MIN,
                                       EavesdropperParameters.EVE_INTERCEPT_PROB_MAX,
                                       float(st.session_state.get('eve_prob', 0.1)),
                                       0.01,
                                       help=h("eve_prob"))
                    st.session_state['eve_prob'] = eve_prob
                else:
                    eve_model = st.session_state.get('eve_model', EavesdropperParameters.EVE_MODEL_DEFAULT)
                    eve_prob = st.session_state.get('eve_prob', EavesdropperParameters.EVE_INTERCEPT_PROB_DEFAULT)

        with col_prev:
            with st.expander("LIVE PREVIEW", expanded=True):
                angle_fig = create_angle_preview([alice_0, alice_1, alice_2], [bob_0, bob_1, bob_2])
                st.pyplot(angle_fig, width='stretch')
                # Use config constants for fallback values in preview
                preview_cfg = ExperimentConfig(
                    enable_fiber_loss=enable_fiber,
                    distance_km_A=distance_A,
                    distance_km_B=distance_B,
                    fiber_loss_dB_per_km=fiber_loss_db if enable_fiber else ChannelLossParameters.FIBER_LOSS_DB_PER_KM_DEFAULT,
                    loss_dB_A=loss_a if enable_fiber else ChannelLossParameters.LOSS_DB_A_DEFAULT,
                    loss_dB_B=loss_b if enable_fiber else ChannelLossParameters.LOSS_DB_B_DEFAULT,
                    enable_repeaters=enable_repeater,
                    num_repeaters=num_repeaters if enable_repeater else ChannelLossParameters.NUM_REPEATERS_DEFAULT,
                    repeater_gain_dB=repeater_gain_dB if enable_repeater else ChannelLossParameters.REPEATER_GAIN_DB_DEFAULT,
                    enable_detector_loss=enable_detector,
                    end_detector_efficiency=end_det_eff,
                    enable_satellite=enable_satellite,
                    slant_range_km=slant_km if enable_satellite else SatelliteParameters.SLANT_RANGE_KM_DEFAULT,
                    transmitter_aperture_m=tx_ap if enable_satellite else SatelliteParameters.TRANSMITTER_APERTURE_M_DEFAULT,
                    receiver_aperture_m=rx_ap if enable_satellite else SatelliteParameters.RECEIVER_APERTURE_M_DEFAULT,
                    beam_divergence_urad=beam_div if enable_satellite else SatelliteParameters.BEAM_DIVERGENCE_URAD_DEFAULT,
                    pointing_jitter_urad=pointing if enable_satellite else SatelliteParameters.POINTING_JITTER_URAD_DEFAULT,
                    receiver_fov_urad=rx_fov if enable_satellite else SatelliteParameters.RECEIVER_FOV_URAD_DEFAULT,
                )
                lb = create_loss_budget_preview(preview_cfg)
                if lb is not None:
                    st.pyplot(lb, width='stretch')
                # Active features summary
                st.markdown("**Active Features:**")
                active = []
                if enable_depol:
                    active.append(f"Depolarizing (alpha={depol_alpha:.3f})")
                if enable_detector:
                    active.append(f"Detector (eta_end={end_det_eff:.2f})")
                if enable_fiber:
                    active.append(f"Fiber loss ({fiber_loss_db:.2f} dB/km)")
                if enable_bg:
                    active.append(f"Background (Y0={Y0:.1e}, cps={bg_cps:.0f})")
                if enable_jitter:
                    active.append(f"Timing jitter ({jitter_ns:.2f} ns)")
                if enable_deadtime:
                    active.append(f"Deadtime ({deadtime_ns:.1f} ns)")
                if enable_after:
                    active.append(f"Afterpulsing (p={after_prob:.3f})")
                if enable_satellite:
                    active.append(f"Satellite (range={slant_km:.0f} km)")
                if enable_multi:
                    active.append(f"SPDC multi-pair (mu={spdc_mu:.2f})")
                if enable_eve and eve_prob > 0:
                    active.append(f"Eve intercept (p={eve_prob:.2f})")
                if not active:
                    st.caption("No additional effects enabled.")
                else:
                    for item in active:
                        st.markdown(f"- {item}")

        st.markdown("---")
        run_button = st.button("Run Experiment", type="primary", help=h("run_experiment"))

        # Build config from UI inputs first
        config = build_experiment_config(
            backend, seed, num_pairs,
            alice_0, alice_1, alice_2, bob_0, bob_1, bob_2,
            bell_state, pA, pB,
            epsilon_sec, epsilon_cor, pe_fraction,
            ir_protocol, f_EC, block_length, code_rate,
            enable_routed_di, di_bound_type, routing_f, routing_d,
            enable_depol, depol_alpha,
            enable_vis, visibility,
            enable_intrinsic, intrinsic_ed,
            enable_fiber, distance_A, distance_B, fiber_loss_db, loss_a, loss_b,
            enable_insertion, insertion_loss,
            enable_misalignment, pol_drift,
            enable_repeater, num_repeaters, repeater_gain_dB,
            enable_detector, herald_eff, end_det_eff,
            enable_dark, use_cps, dark_cps, dark_prob,
            enable_jitter, jitter_ns,
            enable_deadtime, deadtime_ns,
            enable_after, after_prob, after_delay,
            enable_sat, saturation_rate,
            enable_bg, Y0, bg_cps, coinc_ns, rep_rate, enable_timetag,
            enable_multi, spdc_mu, spdc_dist, pair_rate, pump_mw, wavelength, filt_band,
            enable_satellite, beam_div, pointing, rx_fov, slant_km, tx_ap, rx_ap, is_day, sat_bg_cps,
            enable_turb, cn2,
            enable_eve, eve_model, eve_prob,
            preset
        )

        # If a preset is selected, apply it (this will reset ALL parameters to preset values)
        # The preset config will completely override the UI inputs
        if preset != "Custom":
            config = get_preset_config(preset, config)
            # Update current_preset to match
            if st.session_state.current_preset != preset:
                st.session_state.current_preset = preset
        for w in validate_config(config):
            st.warning(w)
        if run_button:
            pbar = st.progress(0)
            status = st.empty()
            def pcb(p, m):
                pbar.progress(p)
                status.text(m)
            try:
                proto = E91Protocol(config)
                results = proto.run(progress_callback=pcb)
                pbar.empty()
                status.empty()
                st.success(f"Experiment completed in {results.execution_time:.2f}s")
                st.session_state.results = results
                st.session_state.config = config
                st.session_state.experiment_history.append({'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"), 'preset': config.preset_name, 'results': results, 'config': config})
                st.rerun()
            except Exception as e:
                st.error(f"Experiment failed: {e}")

        if 'results' in st.session_state and st.session_state.results is not None:
            display_experiment_results(st.session_state.results, st.session_state.config)

    # ========================================================================
    # TAB 3: MESSAGE TEST (Practical QKD Application)
    # ========================================================================
    # Test encryption/decryption with QKD keys + live two-way communication
    # ========================================================================
    with tab_msg:
        st.subheader("Message Test")
        if 'results' not in st.session_state or st.session_state.results is None:
            st.info("Run an experiment to generate keys for testing.")
        else:
            res: ExperimentResults = st.session_state.results
            cfg: ExperimentConfig = st.session_state.config
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Key Bits", f"{len(res.sifted_alice):,}", help="Number of sifted bits available for message testing.")
            with k2:
                st.metric("QBER", f"{res.qber*100:.2f}%", help="Quantum Bit Error Rate of the sifted key.")
            with k3:
                st.metric("CHSH S", f"{res.chsh_S:.3f}", help="Bell parameter; S>2 indicates nonlocality.")
            with k4:
                st.metric("Key Quality", "Good" if res.qber < 0.05 else ("Acceptable" if res.qber < 0.11 else "High"), help="Heuristic label based on QBER.")

            st.markdown("---")

            # Check key quality and availability
            key_available = len(res.sifted_alice) > 0 if hasattr(res, 'sifted_alice') else False
            if not key_available:
                st.error("⚠️ No sifted key bits available! Run an experiment with successful key generation first.")
            elif res.qber > 0.11:
                st.warning(f"⚠️ High QBER ({res.qber*100:.2f}%)! Messages may not be secure. Recommended QBER < 11%.")

            mode = st.radio("Test Mode", ["Self-Test (Loopback)", "Two-Party (Alice -> Bob)"], horizontal=True,
                           help="Self-Test: Encrypt and decrypt with same key (loopback). Two-Party: Test with Alice's and Bob's keys separately.")

            c1, c2 = st.columns([2, 1])
            with c1:
                user_message = st.text_area("Enter message to encrypt", value="Hello, quantum world!", height=100,
                                           help="Your message will be encrypted using One-Time Pad (OTP) with the QKD-generated key",
                                           disabled=not key_available)
            with c2:
                st.markdown("**Message Stats:**")
                st.metric("Characters", len(user_message))
                bits_needed = len(user_message.encode('utf-8'))*8
                st.metric("Bits Required", bits_needed)
                key_length = len(res.sifted_alice) if key_available else 0
                usage = bits_needed/max(1, key_length)*100
                st.metric("Key Usage", f"{usage:.1f}%")

                # Validate key length
                if bits_needed > key_length:
                    st.error(f"⚠️ Message too long! Need {bits_needed} bits, have {key_length} bits.")
                elif usage > 80:
                    st.warning(f"⚠️ High key usage ({usage:.0f}%)! Consider shorter message.")
                elif usage > 0:
                    st.success(f"✅ Sufficient key available")

            # Determine if button should be enabled
            can_encrypt = key_available and len(user_message) > 0 and bits_needed <= key_length

            if st.button("Encrypt & Transmit", type="primary", disabled=not can_encrypt,
                        help="Encrypt the message using One-Time Pad (OTP) with your QKD key and test transmission"):
                try:
                    if mode.startswith("Self-Test"):
                        mt = test_self_message(user_message, res.sifted_alice)
                        st.session_state.message_test = mt
                        st.session_state.message_history.append({
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'mode': mode,
                            'result': mt
                        })
                    else:  # Two-Party mode
                        mt = test_two_party_message(user_message, res.sifted_alice, res.sifted_bob)
                        st.session_state.message_test = mt
                        st.session_state.message_history.append({
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'mode': mode,
                            'result': mt
                        })
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Transmission failed: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

            if 'message_test' in st.session_state:
                st.markdown("---")
                mt = st.session_state.message_test

                # Show success/failure banner
                if mt.success:
                    st.success("✅ MESSAGE TRANSMITTED SUCCESSFULLY!")
                else:
                    st.error(f"❌ TRANSMISSION FAILED: {mt.error_message}")

                d1, d2 = st.columns(2)
                with d1:
                    st.markdown("### 📊 Transmission Details")
                    mode_label = 'Self-Test (Loopback)' if mt.mode == 'self_test' else 'Two-Party (Alice → Bob)'
                    st.write(f"**Mode:** {mode_label}")
                    st.write(f"**Encryption:** One-Time Pad (XOR)")
                    st.write(f"**Key Bits Used:** {mt.key_length_bits:,}")
                    st.write(f"**Message Size:** {len(mt.encrypted_message)} bytes")
                    st.write(f"**Transmission Time:** {mt.transmission_time*1000:.2f} ms")

                    # Show key mismatch info if two-party and failed
                    if mt.mode == 'two_party' and not mt.success and hasattr(mt, 'key_mismatch_positions') and mt.key_mismatch_positions:
                        st.warning(f"⚠️ Key mismatch at {len(mt.key_mismatch_positions)} positions")

                    with st.expander("🔍 View Encrypted Data (Hex)", expanded=False):
                        st.code(format_hex(mt.encrypted_message), language='text')
                        st.caption(f"Total: {len(mt.encrypted_message)} bytes")

                with d2:
                    st.markdown("### ✉️ Message Comparison")
                    st.markdown("**Original:**")
                    st.code(mt.original_message, language=None)

                    st.markdown("**Decrypted:**")
                    if mt.success:
                        st.code(mt.decrypted_message, language=None)
                        if mt.original_message == mt.decrypted_message:
                            st.success("✅ Perfect match!")
                    else:
                        st.code(mt.decrypted_message if mt.decrypted_message else "[Decryption failed]", language=None)
                        if mt.decrypted_message and mt.decrypted_message != mt.original_message:
                            st.error("❌ Mismatch detected!")

                # Show additional info for two-party mode
                if mt.mode == 'two_party':
                    with st.expander("🔬 Advanced Diagnostics", expanded=False):
                        st.markdown("**Alice's Key (first 100 bits):**")
                        alice_preview = mt.alice_key[:100] if hasattr(mt, 'alice_key') and mt.alice_key else []
                        st.text(''.join(map(str, alice_preview)))

                        st.markdown("**Bob's Key (first 100 bits):**")
                        bob_preview = mt.bob_key[:100] if hasattr(mt, 'bob_key') and mt.bob_key else []
                        st.text(''.join(map(str, bob_preview)))

                        if hasattr(mt, 'key_mismatch_positions') and mt.key_mismatch_positions:
                            st.markdown(f"**Mismatch Positions:** {mt.key_mismatch_positions[:20]}...")
                        else:
                            st.success("Keys match perfectly!")

            # ============================================================================
            # ENHANCED NETWORKED TWO-WAY COMMUNICATION (Computer-to-Computer)
            # ============================================================================
            st.markdown("---")
            st.markdown("### 🌐 Live Two-Way Communication (Computer-to-Computer)")
            st.info("**Instructions**: Two users can test live quantum-secured communication between separate machines. Both users must run QKD experiments first to generate shared keys.")

            # Connection Settings
            with st.expander("🔧 Connection Settings", expanded=True):
                net_col1, net_col2 = st.columns([3, 2])
                with net_col1:
                    net_role = st.radio(
                        "Your Role",
                        ["Alice & Bob (Bidirectional)", "Alice (Sender Only)", "Bob (Receiver Only)"],
                        horizontal=True,
                        help="Bidirectional mode allows both sending and receiving on the same machine."
                    )

                    conn_col1, conn_col2 = st.columns(2)
                    with conn_col1:
                        local_port = st.number_input("Your Port",
                                                     min_value=NetworkParameters.PORT_MIN,
                                                     max_value=NetworkParameters.PORT_MAX,
                                                     value=NetworkParameters.DEFAULT_LOCAL_PORT,
                                                     step=1,
                                                     help="Port to listen on for incoming messages")
                    with conn_col2:
                        peer_port = st.number_input("Peer Port",
                                                    min_value=NetworkParameters.PORT_MIN,
                                                    max_value=NetworkParameters.PORT_MAX,
                                                    value=NetworkParameters.DEFAULT_PEER_PORT,
                                                    step=1,
                                                    help="Port where your peer is listening")

                    peer_host = st.text_input("Peer IP Address",
                                              value=NetworkParameters.DEFAULT_PEER_HOST,
                                              help="IP address of the other computer (use 127.0.0.1 for same machine testing)")

                    # Background listener controls
                    listener_col1, listener_col2 = st.columns(2)
                    with listener_col1:
                        if not st.session_state.listener.active:
                            if st.button("🎧 Start Background Listener", type="primary", width='stretch',
                                       help="Start continuously listening for incoming messages in the background"):
                                try:
                                    st.session_state.listener.start(int(local_port))
                                    st.success(f"✅ Listening on port {local_port}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Failed to start listener: {e}")
                                    st.caption("💡 Tip: Port may be in use. Try a different port or restart the application.")
                        else:
                            if st.button("⏹️ Stop Listener", width='stretch',
                                       help="Stop the background listener"):
                                try:
                                    st.session_state.listener.stop()
                                    st.info("✅ Listener stopped")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Failed to stop listener: {e}")

                    with listener_col2:
                        if st.session_state.listener.active:
                            st.success(f"🟢 Listening on :{st.session_state.listener.port}")
                        else:
                            st.info("⚪ Not listening")

                with net_col2:
                    st.markdown("**📋 Quick Setup:**")
                    st.caption("1. Both users run QKD experiments")
                    st.caption("2. Configure ports (must differ for local testing)")
                    st.caption("3. Exchange IP addresses")
                    st.caption("4. Ensure firewall allows UDP traffic")
                    st.caption("5. Start background listener")
                    st.caption("6. Send/receive messages!")

            # Message polling fragment - automatically checks for new messages
            # Uses st.fragment with run_every to poll efficiently without blocking the UI
            @st.fragment(run_every=NetworkParameters.AUTO_REFRESH_INTERVAL_SEC if st.session_state.listener.active else None)
            def message_receiver_fragment():
                """Fragment that polls for new messages from the background listener."""
                if not st.session_state.listener.active or not res:
                    return

                new_messages = st.session_state.listener.get_new_messages()
                if not new_messages:
                    return

                # Determine which key to use based on role
                can_receive = "Bob" in net_role or "Bidirectional" in net_role
                receive_key = res.sifted_bob if hasattr(res, 'sifted_bob') and res.sifted_bob else res.sifted_alice

                if can_receive and receive_key:
                    for msg_info in new_messages:
                        try:
                            lres = listener_receive_message(msg_info['data'], receive_key)
                            st.session_state.message_history.append({
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "mode": "Network Receive",
                                "result": lres,
                                "sender": msg_info['sender']
                            })
                            if lres.success:
                                st.toast(f"📬 New message from {msg_info['sender']}: {lres.decrypted_message[:30]}...", icon="✅")
                        except Exception as e:
                            logging.error(f"Failed to decrypt received message: {e}")
                            st.session_state.message_history.append({
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "mode": "Network Receive (Failed)",
                                "result": MessageTest("listener", "[Unknown]", msg_info['data'], "", [], 0, "One-Time Pad", False, str(e), 0.0),
                                "sender": msg_info['sender']
                            })

            # Call the fragment to activate polling when listener is active
            message_receiver_fragment()

            # Messaging Interface
            st.markdown("---")
            msg_col1, msg_col2 = st.columns([3, 2])

            with msg_col1:
                st.markdown("**✉️ Send Message**")
                net_msg = st.text_area("Message to Send", value="Hello World", height=80,
                                       help="Enter your message here. It will be encrypted with your QKD key.")

                send_col1, send_col2 = st.columns([1, 1])
                with send_col1:
                    can_send = "Alice" in net_role or "Bidirectional" in net_role
                    if st.button("📤 Send Message", type="primary", disabled=not can_send, width='stretch', help="Encrypt and send message to peer via UDP"):
                        if can_send and res and res.sifted_alice:
                            try:
                                mt = test_self_message(net_msg, res.sifted_alice)
                                if not mt.success:
                                    st.error(f"❌ Encryption failed: {mt.error_message}")
                                else:
                                    send_udp_message(peer_host, int(peer_port), mt.encrypted_message)
                                    st.session_state.message_history.append({
                                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "mode": "Network Send",
                                        "result": mt,
                                    })
                                    st.success(f"✅ Sent {len(mt.encrypted_message)} bytes to {peer_host}:{peer_port}")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"❌ Send failed: {e}")
                        elif not res:
                            st.error("❌ Please run QKD experiment first to generate keys")

                with send_col2:
                    st.metric("Messages Sent", len([m for m in st.session_state.message_history if "Send" in m.get("mode", "")]))

            with msg_col2:
                st.markdown("**📊 Communication Status**")
                total_msgs = len([m for m in st.session_state.message_history if "Receive" in m.get("mode", "")])
                st.metric("Messages Received", total_msgs)
                st.metric("Your Port", f":{local_port}")
                st.metric("Peer Address", f"{peer_host}:{peer_port}")

                # Show last received message
                received_msgs = [m for m in st.session_state.message_history if "Receive" in m.get("mode", "")]
                if received_msgs:
                    last_msg = received_msgs[-1]
                    with st.expander("📬 Last Received", expanded=True):
                        if last_msg['result'].success:
                            st.success(f"From: {last_msg.get('sender', 'Unknown')}")
                            st.code(last_msg['result'].decrypted_message, language=None)
                        else:
                            st.error(f"Decryption failed: {last_msg['result'].error_message}")

    # ========================================================================
    # TAB 4: HISTORY (Session Data Review)
    # ========================================================================
    # View and manage experiment, sweep, and message history
    # ========================================================================
    with tab_hist:
        st.subheader("History")
        st.markdown("## Experiment History")
        if st.session_state.experiment_history:
            data = []
            for i, e in enumerate(st.session_state.experiment_history):
                data.append({'Run': i+1, 'Timestamp': e['timestamp'], 'Preset': e['preset'], 'CHSH S': f"{e['results'].chsh_S:.4f}", 'QBER (%)': f"{e['results'].qber*100:.3f}", 'Key Bits': f"{e['results'].num_key_bits:,}", 'Key Rate': f"{e['results'].key_rate_finite:.6f}"})
            st.dataframe(pd.DataFrame(data), width='stretch', hide_index=True)
            a1, a2 = st.columns(2)
            with a1:
                if st.button("Clear Experiment History", width='stretch', help="Delete all saved experiment results from history"):
                    st.session_state.experiment_history = []
                    st.rerun()
            with a2:
                exp_json = json.dumps([{'timestamp': e['timestamp'], 'preset': e['preset'], 'results': asdict(e['results'])} for e in st.session_state.experiment_history], indent=2, default=str)
                st.download_button("Download Experiment History", exp_json, "experiment_history.json", "application/json", width='stretch', help="Export experiment history as JSON file")
        else:
            history_empty_card("No experiments in history yet. Run some experiments to compare results!")
        st.markdown("---")
        st.markdown("## Parametric Sweeps")
        if st.session_state.sweep_history:
            data = []
            for i, e in enumerate(st.session_state.sweep_history):
                data.append({'Sweep': i+1, 'Timestamp': e['timestamp'], 'Parameter': e['parameter'], 'Points': len(e['results'])})
            st.dataframe(pd.DataFrame(data), width='stretch', hide_index=True)
            s1, s2 = st.columns(2)
            with s1:
                if st.button("Clear Sweep History", width='stretch', help="Delete all saved parametric sweep results from history"):
                    st.session_state.sweep_history = []
                    st.rerun()
            with s2:
                sweep_json = json.dumps([{'timestamp': e['timestamp'], 'parameter': e['parameter'], 'num_points': len(e['results'])} for e in st.session_state.sweep_history], indent=2, default=str)
                st.download_button("Download Sweep History", sweep_json, "sweep_history.json", "application/json", width='stretch', help="Export parametric sweep history as JSON file")
        else:
            history_empty_card("No sweep history yet. Run a sweep to populate.")
        st.markdown("---")
        st.markdown("## Messages")
        if st.session_state.message_history:
            data = []
            for i, e in enumerate(st.session_state.message_history):
                data.append({'Message': i+1, 'Timestamp': e['timestamp'], 'Mode': e['mode'], 'Success': bool(e['result'].success), 'Bits Used': e['result'].key_length_bits})
            st.dataframe(pd.DataFrame(data), width='stretch', hide_index=True)
            m1, m2 = st.columns(2)
            with m1:
                if st.button("Clear Message History", width='stretch', help="Delete all saved message test results from history"):
                    st.session_state.message_history = []
                    st.rerun()
            with m2:
                msg_json = json.dumps([{'timestamp': e['timestamp'], 'mode': e['mode'], 'success': e['result'].success, 'key_bits_used': e['result'].key_length_bits} for e in st.session_state.message_history], indent=2, default=str)
                st.download_button("Download Message History", msg_json, "message_history.json", "application/json", width='stretch', help="Export message test history as JSON file")
        else:
            history_empty_card("No messages yet. Use Message Test tab to send one.")
        st.markdown("---")
        st.markdown("## Export All History")
        all_history = {
            'experiments': [{'timestamp': e['timestamp'], 'preset': e['preset'], 'config': asdict(e['config']), 'results': asdict(e['results'])} for e in st.session_state.experiment_history],
            'sweeps': [{'timestamp': e['timestamp'], 'parameter': e['parameter'], 'results': [(val, asdict(res)) for val, res in e['results']]} for e in st.session_state.sweep_history],
            'messages': [{'timestamp': e['timestamp'], 'mode': e['mode'], 'success': e['result'].success, 'key_bits_used': e['result'].key_length_bits, 'transmission_time': e['result'].transmission_time} for e in st.session_state.message_history],
        }
        all_json = json.dumps(all_history, indent=2, default=str)
        st.download_button("Download All (JSON)", all_json, "complete_history.json", "application/json", help="Export complete history (experiments, sweeps, and messages) as single JSON file")


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
