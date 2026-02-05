"""
================================================================================
UI HELPER FUNCTIONS FOR E91 QKD GUI
================================================================================

Helper functions to reduce code duplication in the main Streamlit application.
Includes preset application, config building, and results display.

Author: Tyler Barr
Version: 7.0.0 Modular
Date: 2025

================================================================================
"""

import streamlit as st
import pandas as pd
import json
import io
from typing import List
from dataclasses import asdict

from .models import ExperimentConfig, ExperimentResults
from .config import QuantumConstants
from .visualization import create_results_plots

# Extract constants for easier use
CHSH_CLASSICAL_BOUND = QuantumConstants.CHSH_CLASSICAL_BOUND
QBER_THRESHOLD = QuantumConstants.QBER_THRESHOLD


# ============================================================================
# PRESET APPLICATION
# ============================================================================

def apply_preset_to_session_state(preset: str) -> None:
    """
    Apply a preset configuration to Streamlit session state.

    This function resets ALL session state variables to ensure presets work correctly.
    Matches the comprehensive reset logic in presets.get_preset_config().

    Args:
        preset: Name of the preset to apply
    """
    # ========================================================================
    # SPECIAL CASE: "Custom" preset should preserve user settings
    # ========================================================================
    if preset == "Custom":
        # Don't modify session state - preserve user's custom settings
        return

    # ========================================================================
    # STEP 1: RESET ALL ENABLE FLAGS TO FALSE (IDEAL STATE)
    # ========================================================================
    st.session_state.update({
        'enable_depol': False,
        'enable_vis': False,
        'enable_intrinsic': False,
        'enable_fiber': False,
        'enable_repeater': False,
        'enable_detector': False,
        'enable_dark': False,
        'enable_jitter': False,
        'enable_deadtime': False,
        'enable_after': False,
        'enable_sat': False,
        'enable_bg': False,
        'enable_multi': False,
        'enable_satellite': False,
        'enable_eve': False,
        'enable_routed_di': False,
        'enable_insertion': False,
        'enable_misalign': False,
        'enable_turb': False,
        'enable_timetag': False,
    })

    # ========================================================================
    # STEP 2: SET ALL PARAMETERS TO IDEAL/ZERO VALUES
    # ========================================================================
    import numpy as np
    st.session_state.update({
        # Core parameters
        'num_pairs': 10000,
        'seed': 42,

        # Measurement angles (OPTIMAL CHSH for ideal)
        # Alice: 0°, 22.5°, 45° | Bob: 22.5°, 45°, 67.5°
        'alice_0': 0.0,
        'alice_1': np.pi/8,  # 22.5°
        'alice_2': np.pi/4,  # 45°
        'bob_0': np.pi/8,    # 22.5°
        'bob_1': np.pi/4,    # 45°
        'bob_2': 3*np.pi/8,  # 67.5°

        # Bell state and basis settings
        'bell_state': 'phi_plus',  # Standard E91 Bell state
        'pA': [1/3, 1/3, 1/3],  # Uniform basis probabilities
        'pB': [1/3, 1/3, 1/3],

        # Noise parameters (ZERO for ideal)
        'depol_alpha': 0.0,
        'visibility': 1.0,
        'intrinsic_ed': 0.0,

        # Fiber loss (ZERO for ideal)
        'distance_A': 0.0,
        'distance_B': 0.0,
        'fiber_loss_db': 0.2,
        'loss_a': 0.0,
        'loss_b': 0.0,

        # Insertion/misalignment (ZERO for ideal)
        'insertion_loss': 0.0,
        'polar_drift': 0.0,

        # Repeaters (disabled)
        'num_repeaters': 0,
        'repeater_gain_dB': 0.0,

        # Detector efficiency (PERFECT for ideal)
        'herald_eff': 1.0,
        'end_det_eff': 1.0,

        # Dark counts (ZERO for ideal)
        'use_cps': False,
        'dark_cps': 0.0,
        'dark_prob': 0.0,

        # Timing effects (ZERO for ideal)
        'jitter_ns': 0.0,
        'deadtime_ns': 0.0,
        'after_prob': 0.0,
        'after_delay': 100.0,
        'saturation_rate': 1e6,

        # Background/coincidences (ZERO for ideal)
        'Y0': 0.0,
        'bg_cps': 0.0,
        'coinc_ns': 1.0,
        'rep_rate': 1e6,

        # Multi-pair (disabled)
        'spdc_mu': 0.1,
        'pair_rate': 1e6,
        'pump_mw': 10.0,
        'wavelength': 810.0,
        'filt_band': 3.0,

        # Satellite (disabled)
        'beam_div': 10.0,
        'pointing': 5.0,
        'rx_fov': 100.0,
        'slant_km': 500.0,
        'tx_ap': 0.3,
        'rx_ap': 1.0,
        'is_day': False,
        'sat_bg_cps': 1000.0,
        'cn2': 1e-15,

        # Eavesdropper (disabled)
        'eve_model': "intercept_resend",
        'eve_prob': 0.0,

        # Device-independent (disabled)
        'di_bound_type': "default_di_bound",
        'routing_f': 0.95,
        'routing_d': 0.90,

        # Advanced security parameters
        'epsilon_sec': 1e-9,
        'epsilon_cor': 1e-15,
        'pe_fraction': 0.1,
        'ir_protocol': 'BBBSS',
        'f_EC': 1.2,
        'block_length': 1024,
        'code_rate': 0.5,
    })

    # ========================================================================
    # STEP 3: APPLY PRESET-SPECIFIC CONFIGURATIONS
    # ========================================================================

    if preset == "Ideal (No Losses)":
        # Everything already set to ideal/zero above
        st.session_state['num_pairs'] = 100000  # More pairs for ideal case

    elif preset == "Low Noise (alpha=0.06)":
        st.session_state.update({
            'num_pairs': 25000,  # Increased to ensure enough sifted bits with noise
            'enable_depol': True,
            'depol_alpha': 0.06,
        })

    elif preset == "Moderate Fiber (10km)":
        st.session_state.update({
            'num_pairs': 20000,
            'enable_depol': True,
            'depol_alpha': 0.05,
            'enable_fiber': True,
            'distance_A': 5.0,
            'distance_B': 5.0,
            'fiber_loss_db': 0.2,
            'enable_detector': True,
            'herald_eff': 0.9,
            'end_det_eff': 0.85,
        })

    elif preset == "Long Distance (50km)":
        st.session_state.update({
            'num_pairs': 50000,
            'enable_depol': True,
            'depol_alpha': 0.08,
            'enable_fiber': True,
            'distance_A': 25.0,
            'distance_B': 25.0,
            'fiber_loss_db': 0.2,
            'enable_detector': True,
            'herald_eff': 0.85,
            'end_det_eff': 0.70,
            'enable_repeater': True,
            'num_repeaters': 2,
            'repeater_gain_dB': 3.0,
            'enable_dark': True,
            'dark_prob': 0.002,
        })

    elif preset == "Realistic Lab":
        st.session_state.update({
            'num_pairs': 25000,
            'enable_depol': True,
            'depol_alpha': 0.06,
            'enable_vis': True,
            'visibility': 0.98,
            'enable_fiber': True,
            'distance_A': 2.5,
            'distance_B': 2.5,
            'fiber_loss_db': 0.2,
            'enable_detector': True,
            'herald_eff': 0.9,
            'end_det_eff': 0.85,
            'enable_dark': True,
            'use_cps': True,
            'dark_cps': 100.0,
            'enable_insertion': True,
            'insertion_loss': 0.5,
            'enable_jitter': True,
            'jitter_ns': 0.5,
        })

    elif preset == "Satellite LEO":
        st.session_state.update({
            'num_pairs': 100000,
            'enable_depol': True,
            'depol_alpha': 0.04,
            'enable_satellite': True,
            'slant_km': 300.0,  # Closer pass
            'beam_div': 8.0,  # Tighter beam
            'pointing': 0.5,  # Excellent tracking
            'rx_fov': 100.0,
            'tx_ap': 0.3,
            'rx_ap': 3.0,  # 3m observatory telescope
            'is_day': False,
            'enable_detector': True,
            'end_det_eff': 0.6,  # High-quality SNSPDs
            'herald_eff': 0.95,  # Excellent heralding
            'enable_bg': True,
            'bg_cps': 100.0,  # Low background
        })


# ============================================================================
# CONFIG BUILDING
# ============================================================================

def build_experiment_config(
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
) -> ExperimentConfig:
    """
    Build an ExperimentConfig from all UI parameters.

    Returns:
        ExperimentConfig object with all parameters set
    """
    return ExperimentConfig(
        backend=backend, seed=seed, num_pairs=int(num_pairs),
        alice_angles=[alice_0, alice_1, alice_2], bob_angles=[bob_0, bob_1, bob_2],
        bell_state=bell_state,
        alice_basis_probs=pA, bob_basis_probs=pB,
        epsilon_sec=epsilon_sec, epsilon_cor=epsilon_cor, pe_fraction=float(pe_fraction),
        ir_protocol=ir_protocol, f_EC=float(f_EC), block_length=int(block_length), code_rate=float(code_rate),
        enable_routed_di=enable_routed_di, di_bound_type=di_bound_type,
        routing_efficiency_f=float(routing_f), routing_efficiency_d=float(routing_d),
        enable_depolarizing_noise=enable_depol, depolarizing_alpha=float(depol_alpha),
        enable_visibility_reduction=enable_vis, visibility=float(visibility),
        enable_intrinsic_error=enable_intrinsic, intrinsic_error_ed=float(intrinsic_ed),
        enable_fiber_loss=enable_fiber, distance_km_A=float(distance_A), distance_km_B=float(distance_B),
        fiber_loss_dB_per_km=float(fiber_loss_db) if enable_fiber else 0.2,
        loss_dB_A=float(loss_a) if enable_fiber else 0.0,
        loss_dB_B=float(loss_b) if enable_fiber else 0.0,
        enable_insertion_loss=enable_insertion, insertion_loss_dB=float(insertion_loss),
        enable_misalignment=enable_misalignment, polarization_drift_deg=float(pol_drift),
        enable_repeaters=bool(enable_repeater), num_repeaters=int(num_repeaters) if enable_repeater else 0,
        repeater_gain_dB=float(repeater_gain_dB) if enable_repeater else 0.0,
        enable_detector_loss=enable_detector, heralding_efficiency=float(herald_eff), end_detector_efficiency=float(end_det_eff),
        enable_dark_counts=enable_dark, use_dark_cps=bool(use_cps) if enable_dark else False,
        dark_cps=float(dark_cps) if enable_dark else 100.0, dark_prob=float(dark_prob) if enable_dark else 0.001,
        enable_timing_jitter=enable_jitter, jitter_ns=float(jitter_ns),
        enable_deadtime=enable_deadtime, deadtime_ns=float(deadtime_ns),
        enable_afterpulsing=enable_after, afterpulsing_prob=float(after_prob) if enable_after else 0.05,
        afterpulsing_delay_ns=float(after_delay) if enable_after else 100.0,
        enable_saturation=enable_sat, saturation_rate=float(saturation_rate),
        enable_background=enable_bg, Y0=float(Y0) if enable_bg else 1e-6,
        background_cps=float(bg_cps) if enable_bg else 500.0,
        coincidence_window_ns=float(coinc_ns) if enable_bg else 1.0,
        repetition_rate_Hz=float(rep_rate) if enable_bg else 1e6,
        enable_time_tagging=enable_timetag if enable_bg else False,
        enable_multi_pair=enable_multi, spdc_brightness_mu=float(spdc_mu) if enable_multi else 0.1,
        spdc_distribution=spdc_dist if enable_multi else "thermal",
        pair_rate=float(pair_rate) if enable_multi else 1e6,
        pump_power_mW=float(pump_mw) if enable_multi else 10.0,
        wavelength_nm=float(wavelength) if enable_multi else 810.0,
        filter_bandwidth_nm=float(filt_band) if enable_multi else 3.0,
        enable_satellite=enable_satellite, beam_divergence_urad=float(beam_div) if enable_satellite else 10.0,
        pointing_jitter_urad=float(pointing) if enable_satellite else 5.0,
        receiver_fov_urad=float(rx_fov) if enable_satellite else 100.0,
        slant_range_km=float(slant_km) if enable_satellite else 500.0,
        transmitter_aperture_m=float(tx_ap) if enable_satellite else 0.3,
        receiver_aperture_m=float(rx_ap) if enable_satellite else 1.0,
        is_daytime=bool(is_day) if enable_satellite else False,
        satellite_background_cps=float(sat_bg_cps) if enable_satellite else 1000.0,
        enable_turbulence=enable_turb if enable_satellite else False,
        cn2=float(cn2) if enable_satellite and enable_turb else 1e-15,
        enable_eavesdropper=enable_eve, eve_model=eve_model if enable_eve else "intercept_resend",
        eve_intercept_prob=float(eve_prob) if enable_eve else 0.0,
        preset_name=preset,
    )


# ============================================================================
# RESULTS DISPLAY
# ============================================================================

def display_experiment_results(results: ExperimentResults, config: ExperimentConfig) -> None:
    """
    Display experiment results with metrics, plots, and detailed information.

    Args:
        results: ExperimentResults object
        config: ExperimentConfig object used for the experiment
    """
    st.markdown("---")
    st.markdown("### Results")

    # Main metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(
            "CHSH S",
            f"{results.chsh_S:.4f}",
            f"{results.chsh_S-CHSH_CLASSICAL_BOUND:+.4f}",
            help="CHSH Bell parameter. Classical limit is 2.0; quantum max is 2.828. Values >2 indicate nonlocality."
        )
    with c2:
        st.metric(
            "QBER",
            f"{results.qber*100:.3f}%",
            "Secure" if results.qber < QBER_THRESHOLD else "High",
            help="Quantum Bit Error Rate of the sifted key. Acceptable <11%."
        )
    with c3:
        st.metric(
            "Detection eta",
            f"{results.detection_efficiency*100:.2f}%",
            help="Detected/generated fraction after all losses and detector efficiencies."
        )
    with c4:
        st.metric(
            "Sifting eta",
            f"{results.sifting_efficiency*100:.2f}%",
            help="Sifted/detected fraction. E91 ideal ~22.2% with uniform bases and A0B0/A2B2 as key."
        )
    with c5:
        st.metric(
            "Final Key",
            f"{results.num_key_bits:,}",
            help="Usable secret bits after error correction and privacy amplification."
        )

    if getattr(results, 'eve_intercepted', None) is not None:
        st.caption(f"Eve intercepted ~{results.eve_intercepted} pairs (model: intercept_resend)")

    # Results plots
    fig = create_results_plots(config, results)
    st.pyplot(fig)

    # Plot guide
    with st.expander("What do these plots show?", expanded=False):
        st.markdown(
            "- CHSH bar: measured S vs classical 2.0 and quantum 2.828. S>2 indicates nonlocality.\n"
            "- QBER bar: error rate with a threshold at 11%.\n"
            "- Key pipeline: generated -> detected -> sifted -> final key.\n"
            "- Secret fractions and key rates: expected secret bits per sifted/generated bit.\n"
            "- Correlators heatmap: E_ij across all angle combinations; structure reflects settings.\n"
            "- Angles: polar view of Alice/Bob bases; helps confirm CHSH geometry."
        )

    # Metrics guide
    with st.expander("What do these metrics mean?", expanded=False):
        s_secure = "violation (>2)" if results.chsh_S > CHSH_CLASSICAL_BOUND else "no violation"
        q_quality = ("excellent (<5%)" if results.qber < 0.05 else ("acceptable (<11%)" if results.qber < 0.11 else "too high (>=11%)"))
        st.markdown(
            f"- CHSH S: range 0..2.828; classical bound 2.0. Your result {results.chsh_S:.4f} -> {s_secure}.\n"
            f"- QBER: fraction of bit mismatches. Your result {results.qber*100:.3f}% -> {q_quality}.\n"
            f"- Detection efficiency: detected/generated = {results.detection_efficiency*100:.2f}%.\n"
            f"- Sifting efficiency: sifted/detected = {results.sifting_efficiency*100:.2f}%. Ideal ~22.2% for classic E91 mapping.\n"
            f"- Final key bits: {results.num_key_bits:,}. Depends on secret fraction, block sizes, finite-key penalties."
        )

    # Security assessment
    st.markdown("---")
    st.markdown("### Security Assessment")
    secure = (results.chsh_S > CHSH_CLASSICAL_BOUND) and (results.qber < QBER_THRESHOLD)
    if secure:
        st.success(f"SECURE: CHSH={results.chsh_S:.4f} (> {CHSH_CLASSICAL_BOUND}), QBER={results.qber*100:.2f}% (< {QBER_THRESHOLD*100:.0f}%).")
    else:
        tip = "reduce losses or improve detector efficiency" if results.detection_efficiency < 0.5 else "reduce noise (α,e_d) or adjust angles"
        st.error(f"INSECURE: CHSH={results.chsh_S:.4f}, QBER={results.qber*100:.2f}% — {tip}.")

    # Detailed results table
    with st.expander("📄 Detailed Results", expanded=False):
        rows = [
            ("Generated Pairs", f"{results.num_pairs_generated:,}"),
            ("Detected Pairs", f"{results.num_pairs_detected:,}"),
            ("Detection Efficiency", f"{results.detection_efficiency*100:.2f}%"),
            ("Sifted Bits", f"{results.num_sifted:,}"),
            ("Sifting Efficiency", f"{results.sifting_efficiency*100:.2f}%"),
            ("QBER", f"{results.qber*100:.3f}%"),
            ("CHSH S", f"{results.chsh_S:.4f}"),
            ("Secret Fraction (Asymp)", f"{results.secret_fraction_asymptotic:.6f}"),
            ("Secret Fraction (Finite)", f"{results.secret_fraction_finite:.6f}"),
            ("Key Rate (Asymp)", f"{results.key_rate_asymptotic:.6f}"),
            ("Key Rate (Finite)", f"{results.key_rate_finite:.6f}"),
            ("Final Key Bits", f"{results.num_key_bits:,}"),
            ("Execution Time", f"{results.execution_time:.2f}s"),
        ]
        df = pd.DataFrame(rows, columns=["Metric", "Value"])
        st.dataframe(df, width='stretch', hide_index=True)

    # Helper function for bit string formatting
    def _bits_to_str(bits: List[int], group: int = 8, max_len: int = 4096) -> str:
        if not bits:
            return ""
        s = "".join("1" if b else "0" for b in bits[:max_len])
        if group > 0:
            s = " ".join(s[i:i+group] for i in range(0, len(s), group))
        if len(bits) > max_len:
            s += " …"
        return s

    # Raw arrays
    with st.expander("📎 Raw Arrays (bits / time tags)", expanded=False):
        st.markdown("Key bits are shown after sifting. Groups of 8 for readability.")
        st.code(f"Alice sifted: {_bits_to_str(results.sifted_alice)}", language="text")
        st.code(f"Bob sifted:   {_bits_to_str(results.sifted_bob)}", language="text")
        if results.time_tags_alice is not None and results.time_tags_bob is not None:
            ttA = pd.Series(results.time_tags_alice, name="t_A (ns)")
            ttB = pd.Series(results.time_tags_bob, name="t_B (ns)")
            st.caption(f"Time tags recorded: {len(ttA)} events")
            cta, ctb = st.columns(2)
            with cta:
                st.dataframe(ttA.head(100), width='stretch')
            with ctb:
                st.dataframe(ttB.head(100), width='stretch')
            buf_tt = io.StringIO()
            pd.DataFrame({"t_A_ns": ttA, "t_B_ns": ttB}).to_csv(buf_tt, index=False)
            st.download_button("Download Time Tags (CSV)", buf_tt.getvalue(), "time_tags.csv", "text/csv")

    # Download buttons
    st.markdown("---")
    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button("Download Config (JSON)", json.dumps(asdict(config), indent=2, default=str), "config.json", "application/json")
    with e2:
        st.download_button("Download Results (JSON)", json.dumps(asdict(results), indent=2, default=str), "results.json", "application/json")
    with e3:
        if results.sifted_alice:
            df = pd.DataFrame({'alice': results.sifted_alice, 'bob': results.sifted_bob, 'match': [a == b for a, b in zip(results.sifted_alice, results.sifted_bob)]})
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            st.download_button("Download Sifted (CSV)", buf.getvalue(), "sifted.csv", "text/csv")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'apply_preset_to_session_state',
    'build_experiment_config',
    'display_experiment_results',
]
