"""
================================================================================
UTILITY FUNCTIONS FOR E91 QKD
================================================================================

Mathematical utilities, data formatting, and UI helper functions.

Author: Tyler Barr
Version: 7.0.0 Modular
Date: 2025

================================================================================
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from typing import List, Dict
from .models import ExperimentConfig


# ============================================================================
# MATHEMATICAL UTILITIES
# ============================================================================

def normalize_probs(values: List[float]) -> List[float]:
    """
    Normalize a list of non-negative values to sum to 1.
    Falls back safely if the sum is ~0.

    Args:
        values: List of non-negative values

    Returns:
        Normalized list summing to 1.0
    """
    s = float(sum(values))
    if s <= 1e-12:
        n = max(len(values), 1)
        return [1.0 / n] * n
    return [max(0.0, float(v)) / s for v in values]


# ============================================================================
# DATA FORMATTING
# ============================================================================

def format_hex(data: bytes, group: int = 2, line: int = 32) -> str:
    """
    Pretty-format bytes as hex with grouping and line wrapping.

    Args:
        data: Bytes to format
        group: Number of hex chars per group (2 -> byte spacing)
        line: Number of groups per line

    Returns:
        Formatted hex string
    """
    hex_str = data.hex()
    pairs = [hex_str[i : i + group] for i in range(0, len(hex_str), group)]
    lines = [" ".join(pairs[i : i + line]) for i in range(0, len(pairs), line)]
    return "\n".join(lines)


# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================

def history_empty_card(message: str):
    """
    Render a subtle empty-state card used in History sections.

    Args:
        message: Message to display in the card
    """
    st.markdown(
        f"""
        <div style="background:#0f2438;border-radius:10px;padding:14px;color:#cde3ff;">
        {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# HELP TEXT / TOOLTIPS
# ============================================================================

# Centralized help text for UI widgets. Use with help=h("key").
HELP_TEXT: Dict[str, str] = {
    "preset": "Preset bundles of parameters. Choose 'Custom' for full control.",
    "backend": "Quantum simulator backend. 'qiskit' uses circuit simulation; 'qutip' uses analytic probabilities.",
    "seed": "Random seed for reproducibility. Same seed + config -> identical results. Typical range: 0-9999.",
    "total_distance": "Total fiber distance in km. Split equally between Alice and Bob for loss.",
    "angle_type": "Angle input mode. 'Exact' allows any value; 'Whole Radians (pi/n)' snaps to multiples of pi/8.",
    "randomize_angles": "Randomize all measurement angles using the current angle mode.",
    "angle": "Measurement angle in radians (A0..A2, B0..B2). Relative differences set CHSH.",
    "num_pairs": "Number of entangled pairs to simulate. Larger -> better statistics, longer runtime.",
    "pA": "Alice basis probabilities for A0/A1/A2. Must sum to 1.",
    "pB": "Bob basis probabilities for B0/B1/B2. Must sum to 1.",
    "prob_randomize": "Randomize and normalize basis probabilities.",
    "prob_reset": "Reset basis probabilities to defaults: A=[0.5,0.5,0.0], B=[0.5,0.5,0.0].",
    "enable_depol": "Enable depolarizing noise (uniform random Pauli errors).",
    "depol_alpha": "Depolarizing strength alpha in [0,1]. Higher alpha raises QBER and lowers CHSH.",
    "enable_visibility": "Enable visibility reduction (imperfect interference/alignment).",
    "visibility": "Visibility V in [0.5,1]. Secret fraction degrades with (1-V).",
    "enable_intrinsic": "Enable intrinsic bit-flip errors.",
    "intrinsic_ed": "Intrinsic error e_d: probability of a flip per bit.",
    "enable_fiber": "Enable fiber attenuation and fixed losses.",
    "fiber_loss_db": "Fiber attenuation in dB per km (e.g., 0.2 dB/km).",
    "fixed_loss_a": "Fixed loss at Alice (dB).",
    "fixed_loss_b": "Fixed loss at Bob (dB).",
    "enable_detector": "Include heralding and end-detector efficiencies.",
    "herald_eff": "Heralding efficiency in [0,1].",
    "end_det_eff": "End detector efficiency in [0,1].",
    "enable_dark": "Include dark counts.",
    "use_cps": "Specify dark counts in CPS and convert to per-pulse using repetition rate.",
    "dark_cps": "Dark counts per second (CPS).",
    "dark_prob": "Dark probability per pulse when CPS mode is off.",
    "enable_jitter": "Include Gaussian timing jitter on time tags.",
    "jitter_ns": "Timing jitter standard deviation (ns).",
    "enable_deadtime": "Apply detector deadtime after a click.",
    "deadtime_ns": "Deadtime window (ns).",
    "enable_after": "Include afterpulsing events.",
    "after_prob": "Afterpulse probability per detection.",
    "after_delay": "Typical afterpulse delay (ns).",
    "enable_sat": "Apply saturation limit when repetition rate exceeds detector capability.",
    "saturation_rate": "Max sustainable event rate (Hz).",
    "enable_bg": "Include uniform background/stray counts.",
    "Y0": "Background click probability per pulse.",
    "bg_cps": "Background rate in counts per second.",
    "coinc_ns": "Coincidence window width (ns).",
    "rep_rate": "Pulse repetition rate (Hz).",
    "enable_timetag": "Record time tags for coincidence analysis and CAR.",
    "enable_multi": "Enable multi-pair SPDC emission (raises accidentals).",
    "spdc_mu": "SPDC brightness mu: mean pairs per pulse.",
    "pair_rate": "Mean pair production rate (Hz).",
    "pump_mw": "Pump laser power (mW).",
    "wavelength": "Photon wavelength (nm).",
    "filt_band": "Spectral filter bandwidth (nm).",
    "enable_satellite": "Free-space link model (geometric/pointing loss, turbulence optional).",
    "beam_div": "Transmitter beam divergence (microrad).",
    "pointing": "Pointing jitter (microrad).",
    "rx_fov": "Receiver field of view (microrad).",
    "slant_km": "Slant range (km).",
    "tx_ap": "Transmitter aperture (m).",
    "rx_ap": "Receiver aperture (m).",
    "is_day": "Daytime operation increases background.",
    "sat_bg_cps": "Satellite background rate (CPS).",
    "enable_turb": "Enable atmospheric turbulence with structure constant Cn2.",
    "cn2": "Refractive index structure constant Cn2 (m^-2/3).",
    "epsilon_sec": "Security failure probability for privacy amplification.",
    "epsilon_cor": "Correctness failure probability (post-IR).",
    "pe_fraction": "Fraction of sifted bits used for parameter estimation.",
    "ir_protocol": "Information reconciliation protocol (BBBSS or Cascade).",
    "f_EC": "Error-correction efficiency multiplier vs Shannon limit.",
    "block_length": "Block length for IR/PA (bits).",
    "code_rate": "Privacy amplification code rate (0-1).",
    "enable_routed_di": "Compute a simple routed DI witness.",
    "di_bound_type": "Device-independent bound selector (placeholder).",
    "routing_f": "Fixed routing efficiency eta_f.",
    "routing_d": "Distance-dependent routing efficiency eta_d.",
    "enable_eve": "Enable a simple intercept-resend eavesdropper.",
    "eve_model": "Eavesdropper attack model.",
    "eve_prob": "Fraction of pairs intercepted by Eve (0-1).",
    "run_experiment": "Run the simulator with current parameters.",
}


def h(key: str) -> str:
    """
    Lookup helper for tooltip text.

    Args:
        key: Help text key

    Returns:
        Help text string or empty string if not found
    """
    return HELP_TEXT.get(key, "")


# ============================================================================
# SATELLITE CALCULATIONS
# ============================================================================

def compute_satellite_loss(config: ExperimentConfig) -> float:
    """
    Compute satellite channel loss.

    Args:
        config: Experiment configuration

    Returns:
        Total loss in dB
    """
    if not config.enable_satellite:
        return 0.0
    wavelength_m = config.wavelength_nm * 1e-9
    range_m = config.slant_range_km * 1000
    divergence_rad = config.beam_divergence_urad * 1e-6
    beam_radius_m = range_m * divergence_rad
    beam_area = np.pi * beam_radius_m**2
    rx_area = np.pi * (config.receiver_aperture_m / 2)**2
    geometric_loss = min(1.0, rx_area / max(beam_area, 1e-12))
    pointing_error_rad = config.pointing_jitter_urad * 1e-6
    pointing_loss = np.exp(-2 * (pointing_error_rad / divergence_rad)**2)
    turbulence_loss = 1.0
    if config.enable_turbulence:
        k = 2 * np.pi / wavelength_m
        sigma_R2 = 1.23 * config.cn2 * k**(7/6) * range_m**(11/6)
        turbulence_loss = np.exp(-sigma_R2)
    total_efficiency = geometric_loss * pointing_loss * turbulence_loss
    return -10 * np.log10(max(total_efficiency, 1e-10))


# ============================================================================
# RUNTIME ESTIMATION
# ============================================================================

def estimate_runtime(num_pairs: int, backend: str) -> float:
    """
    Estimate runtime in seconds.

    Args:
        num_pairs: Number of entangled pairs
        backend: Backend name ('qiskit' or 'qutip')

    Returns:
        Estimated runtime in seconds
    """
    return num_pairs * (0.0002 if backend == "qiskit" else 0.00005)


# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def validate_config(config: ExperimentConfig) -> List[str]:
    """
    Validate configuration and return warnings.

    Args:
        config: Experiment configuration

    Returns:
        List of warning strings
    """
    warnings = []
    if config.enable_fiber_loss:
        total_loss = (config.distance_km_A + config.distance_km_B) * config.fiber_loss_dB_per_km
        if total_loss > 30:
            warnings.append(f"⚠️ Very high fiber loss ({total_loss:.1f} dB).")
    if config.enable_depolarizing_noise and config.depolarizing_alpha > 0.15:
        warnings.append("⚠️ High depolarizing noise may prevent key generation.")
    if config.num_pairs < 5000:
        warnings.append("⚠️ Low number of pairs. Results may be unreliable.")
    if config.enable_detector_loss and config.end_detector_efficiency < 0.3:
        warnings.append("⚠️ Very low detector efficiency.")
    if config.enable_satellite and config.enable_fiber_loss:
        warnings.append("ℹ️ Both satellite and fiber loss enabled.")
    return warnings


# ============================================================================
# VISUALIZATION PREVIEWS
# ============================================================================

def create_angle_preview(alice_angles: List[float], bob_angles: List[float]):
    """
    Create live preview of measurement angles.

    Args:
        alice_angles: Alice's measurement angles (radians)
        bob_angles: Bob's measurement angles (radians)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(projection='polar'))
    for i, angle in enumerate(alice_angles):
        ax.plot([angle, angle], [0, 0.8], 'b-', linewidth=2, alpha=0.7)
        ax.plot(angle, 0.8, 'bo', markersize=8)
        ax.text(angle, 0.9, f'A{i}', ha='center', fontsize=9, color='blue', fontweight='bold')
    for i, angle in enumerate(bob_angles):
        ax.plot([angle, angle], [0, 0.6], 'r--', linewidth=2, alpha=0.7)
        ax.plot(angle, 0.6, 'rs', markersize=8)
        ax.text(angle, 0.7, f'B{i}', ha='center', fontsize=9, color='red', fontweight='bold')
    ax.set_title('Measurement Angles', fontsize=10, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    return fig


def create_loss_budget_preview(config: ExperimentConfig):
    """
    Create preview of expected loss budget.

    Args:
        config: Experiment configuration

    Returns:
        Matplotlib figure or None if no losses enabled
    """
    loss_components, loss_values, colors = [], [], []
    if config.enable_fiber_loss:
        total_fiber = config.distance_km_A + config.distance_km_B
        fiber_loss = total_fiber * config.fiber_loss_dB_per_km
        # Apply simple repeater compensation in the preview if enabled
        if getattr(config, "enable_repeaters", False) and getattr(config, "num_repeaters", 0) > 0:
            fiber_loss = max(0.0, fiber_loss - config.num_repeaters * config.repeater_gain_dB)
        loss_components.append('Fiber')
        loss_values.append(fiber_loss)
        colors.append('orange')
    if config.enable_insertion_loss:
        loss_components.append('Insertion')
        loss_values.append(config.insertion_loss_dB)
        colors.append('coral')
    if config.enable_detector_loss:
        det_loss = -10 * np.log10(config.end_detector_efficiency)
        loss_components.append('Detector')
        loss_values.append(det_loss)
        colors.append('red')
    if config.enable_satellite:
        sat_loss = compute_satellite_loss(config)
        loss_components.append('Satellite')
        loss_values.append(sat_loss)
        colors.append('purple')
    if not loss_components:
        return None
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(loss_components, loss_values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Loss (dB)', fontsize=9, fontweight='bold')
    ax.set_title('Expected Loss Budget', fontsize=10, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    total_loss = sum(loss_values)
    ax.text(0.95, 0.05, f'Total: {total_loss:.1f} dB', transform=ax.transAxes, ha='right',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    return fig


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'normalize_probs',
    'format_hex',
    'history_empty_card',
    'HELP_TEXT',
    'h',
    'compute_satellite_loss',
    'estimate_runtime',
    'validate_config',
    'create_angle_preview',
    'create_loss_budget_preview',
]
