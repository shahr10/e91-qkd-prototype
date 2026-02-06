"""
================================================================================
FIXED CONFIGURATION PRESETS FOR E91 QKD
================================================================================

Comprehensive preset system that resets ALL 88 parameters.

Author: Tyler Barr
Version: 7.0.1 Fixed
Date: 2025

================================================================================
"""

from .models import ExperimentConfig
import numpy as np


# ============================================================================
# COMPLETE PRESET CONFIGURATIONS
# ============================================================================

def get_preset_config(preset_name: str, base_config: ExperimentConfig) -> ExperimentConfig:
    """
    Load preset configuration with COMPLETE parameter reset.

    This function resets ALL 88 parameters to ensure presets work correctly.

    Args:
        preset_name: Name of the preset to load
        base_config: Base configuration (will be modified)

    Returns:
        Modified configuration with ALL parameters set correctly
    """

    # ========================================================================
    # SPECIAL CASE: "Custom" preset should preserve user settings
    # ========================================================================
    if preset_name == "Custom":
        base_config.preset_name = "Custom"
        return base_config

    # ========================================================================
    # STEP 1: RESET ALL ENABLE FLAGS TO FALSE (IDEAL STATE)
    # ========================================================================

    base_config.enable_depolarizing_noise = False
    base_config.enable_visibility_reduction = False
    base_config.enable_intrinsic_error = False
    base_config.enable_fiber_loss = False
    base_config.enable_insertion_loss = False
    base_config.enable_misalignment = False
    base_config.enable_repeaters = False
    base_config.enable_detector_loss = False
    base_config.enable_dark_counts = False
    base_config.enable_timing_jitter = False
    base_config.enable_deadtime = False
    base_config.enable_afterpulsing = False
    base_config.enable_saturation = False
    base_config.enable_background = False
    base_config.enable_time_tagging = False
    base_config.enable_multi_pair = False
    base_config.enable_satellite = False
    base_config.enable_turbulence = False
    base_config.enable_eavesdropper = False
    base_config.enable_routed_di = False

    # ========================================================================
    # STEP 2: SET ALL PARAMETERS TO IDEAL/ZERO VALUES
    # ========================================================================

    # Core parameters (keep reasonable defaults)
    base_config.num_pairs = 10000
    base_config.seed = 42

    # Measurement angles (OPTIMAL CHSH configuration for all presets)
    # Alice: 0°, 22.5°, 45° | Bob: 22.5°, 45°, 67.5°
    # With CHSH indices (0,2), this gives Tsirelson bound S ≈ 2.828
    base_config.alice_angles = [0.0, np.pi/8, np.pi/4]  # 0°, 22.5°, 45°
    base_config.bob_angles = [np.pi/8, np.pi/4, 3*np.pi/8]  # 22.5°, 45°, 67.5°
    base_config.chsh_a0_idx = 0  # Alice 0°
    base_config.chsh_a1_idx = 2  # Alice 45°
    base_config.chsh_b0_idx = 0  # Bob 22.5°
    base_config.chsh_b1_idx = 2  # Bob 67.5°

    # Bell state and basis probabilities
    base_config.bell_state = "phi_plus"  # Standard E91 entangled state
    base_config.alice_basis_probs = None  # Uniform random selection
    base_config.bob_basis_probs = None    # Uniform random selection

    # Noise parameters (ZERO for ideal)
    base_config.depolarizing_alpha = 0.0
    base_config.visibility = 1.0
    base_config.intrinsic_error_ed = 0.0

    # Fiber loss (ZERO for ideal)
    base_config.distance_km_A = 0.0
    base_config.distance_km_B = 0.0
    base_config.fiber_loss_dB_per_km = 0.2  # Standard value when disabled
    base_config.loss_dB_A = 0.0
    base_config.loss_dB_B = 0.0

    # Insertion/misalignment (ZERO for ideal)
    base_config.insertion_loss_dB = 0.0
    base_config.polarization_drift_deg = 0.0

    # Repeaters (disabled)
    base_config.num_repeaters = 0
    base_config.repeater_gain_dB = 0.0

    # Detector efficiency (PERFECT for ideal)
    base_config.heralding_efficiency = 1.0
    base_config.end_detector_efficiency = 1.0

    # Dark counts (ZERO for ideal)
    base_config.use_dark_cps = False
    base_config.dark_cps = 0.0
    base_config.dark_prob = 0.0

    # Timing effects (ZERO for ideal)
    base_config.jitter_ns = 0.0
    base_config.deadtime_ns = 0.0
    base_config.afterpulsing_prob = 0.0
    base_config.afterpulsing_delay_ns = 100.0
    base_config.saturation_rate = 1e6

    # Background/coincidences (ZERO for ideal)
    base_config.Y0 = 0.0
    base_config.background_cps = 0.0
    base_config.coincidence_window_ns = 1.0
    base_config.repetition_rate_Hz = 1e6

    # Multi-pair (disabled)
    base_config.spdc_brightness_mu = 0.1
    base_config.pair_rate = 1e6
    base_config.pump_power_mW = 10.0
    base_config.wavelength_nm = 810.0
    base_config.filter_bandwidth_nm = 3.0

    # Satellite (disabled)
    base_config.beam_divergence_urad = 10.0
    base_config.pointing_jitter_urad = 5.0
    base_config.receiver_fov_urad = 100.0
    base_config.slant_range_km = 500.0
    base_config.transmitter_aperture_m = 0.3
    base_config.receiver_aperture_m = 1.0
    base_config.is_daytime = False
    base_config.satellite_background_cps = 1000.0
    base_config.cn2 = 1e-15

    # Eavesdropper (disabled)
    base_config.eve_model = "intercept_resend"
    base_config.eve_intercept_prob = 0.0

    # Device-independent (disabled)
    base_config.di_bound_type = "default_di_bound"
    base_config.routing_efficiency_f = 0.95
    base_config.routing_efficiency_d = 0.90

    # ========================================================================
    # STEP 3: APPLY PRESET-SPECIFIC CONFIGURATIONS
    # ========================================================================

    if preset_name == "Tsirelson Optimal (CHSH Max)":
        # Tsirelson bound: maximal CHSH violation S ≈ 2√2 ≈ 2.828
        # Angles already set to optimal CHSH configuration in STEP 2
        base_config.num_pairs = 100000  # Large sample for precision

    elif preset_name == "Ideal (No Losses)":
        # Everything already set to ideal/zero above (including optimal CHSH angles)
        base_config.num_pairs = 100000  # More pairs for ideal case

    elif preset_name == "Low Noise (alpha=0.06)":
        base_config.num_pairs = 25000  # Increased to ensure enough sifted bits with noise
        base_config.enable_depolarizing_noise = True
        base_config.depolarizing_alpha = 0.06

    elif preset_name == "Moderate Fiber (10km)":
        base_config.num_pairs = 20000
        base_config.enable_depolarizing_noise = True
        base_config.depolarizing_alpha = 0.05
        base_config.enable_fiber_loss = True
        base_config.distance_km_A = 5.0
        base_config.distance_km_B = 5.0
        base_config.fiber_loss_dB_per_km = 0.2
        base_config.enable_detector_loss = True
        base_config.heralding_efficiency = 0.9
        base_config.end_detector_efficiency = 0.85

    elif preset_name == "Long Distance (50km)":
        base_config.num_pairs = 50000
        base_config.enable_depolarizing_noise = True
        base_config.depolarizing_alpha = 0.08
        base_config.enable_fiber_loss = True
        base_config.distance_km_A = 25.0
        base_config.distance_km_B = 25.0
        base_config.fiber_loss_dB_per_km = 0.2
        base_config.enable_detector_loss = True
        base_config.heralding_efficiency = 0.85
        base_config.end_detector_efficiency = 0.70
        base_config.enable_repeaters = True
        base_config.num_repeaters = 2
        base_config.repeater_gain_dB = 3.0
        base_config.enable_dark_counts = True
        base_config.dark_prob = 0.002

    elif preset_name == "Realistic Lab":
        base_config.num_pairs = 25000
        base_config.enable_depolarizing_noise = True
        base_config.depolarizing_alpha = 0.06
        base_config.enable_visibility_reduction = True
        base_config.visibility = 0.98
        base_config.enable_fiber_loss = True
        base_config.distance_km_A = 2.5
        base_config.distance_km_B = 2.5
        base_config.fiber_loss_dB_per_km = 0.2
        base_config.enable_detector_loss = True
        base_config.heralding_efficiency = 0.9
        base_config.end_detector_efficiency = 0.85
        base_config.enable_dark_counts = True
        base_config.use_dark_cps = True
        base_config.dark_cps = 100.0
        base_config.enable_insertion_loss = True
        base_config.insertion_loss_dB = 0.5
        base_config.enable_timing_jitter = True
        base_config.jitter_ns = 0.5

    elif preset_name == "Satellite LEO":
        # Satellite QKD from Low Earth Orbit (shorter pass, optimal geometry)
        # Entangled source on satellite, two ground receivers
        # Reduced range + larger receivers for better efficiency (~5% total)
        base_config.num_pairs = 100000  # Sufficient for ~5% detection efficiency
        base_config.enable_depolarizing_noise = True
        base_config.depolarizing_alpha = 0.04
        base_config.enable_satellite = True
        base_config.slant_range_km = 300.0  # Closer pass (300 km vs 500 km)
        base_config.beam_divergence_urad = 8.0  # Tighter beam (8 μrad)
        base_config.pointing_jitter_urad = 0.5  # Excellent tracking
        base_config.receiver_fov_urad = 100.0
        base_config.transmitter_aperture_m = 0.3  # 30 cm satellite telescope
        base_config.receiver_aperture_m = 3.0  # 3 m ground telescope (large observatory)
        base_config.is_daytime = False  # Night pass for lower background
        base_config.enable_detector_loss = True
        base_config.end_detector_efficiency = 0.6  # High-quality SNSPDs
        base_config.heralding_efficiency = 0.95  # Excellent heralding
        base_config.enable_background = True
        base_config.background_cps = 100.0  # Low background (night + filtering)

    elif preset_name == "Security Test (Eve Present)":
        # Eavesdropper present to demonstrate security degradation
        base_config.num_pairs = 20000
        base_config.enable_depolarizing_noise = True
        base_config.depolarizing_alpha = 0.08
        base_config.enable_fiber_loss = True
        base_config.distance_km_A = 5.0
        base_config.distance_km_B = 5.0
        base_config.fiber_loss_dB_per_km = 0.2
        base_config.enable_detector_loss = True
        base_config.heralding_efficiency = 0.9
        base_config.end_detector_efficiency = 0.85
        base_config.enable_eavesdropper = True
        base_config.eve_model = "intercept_resend"
        base_config.eve_intercept_prob = 0.25

    # Set preset name
    base_config.preset_name = preset_name
    return base_config


# ============================================================================
# PRESET NAMES
# ============================================================================

PRESET_NAMES = [
    "Ideal (No Losses)",
    "Low Noise (alpha=0.06)",
    "Moderate Fiber (10km)",
    "Long Distance (50km)",
    "Realistic Lab",
    "Satellite LEO",
    "Security Test (Eve Present)",
    "Custom"
]


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'get_preset_config',
    'PRESET_NAMES',
]
