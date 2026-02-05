"""
Configuration validator for E91 QKD simulation.

Validates ExperimentConfig for consistency and prevents impossible configurations.

Author: E91 QKD Simulation Team
Date: 2025-12-29
"""

from typing import List, Tuple
from .models import ExperimentConfig
import warnings


def validate_config(config: ExperimentConfig) -> Tuple[List[str], List[str]]:
    """
    Validate ExperimentConfig for consistency.

    Args:
        config: ExperimentConfig to validate

    Returns:
        Tuple of (errors, warnings)
        - errors: Fatal issues that prevent running
        - warnings: Non-fatal issues user should know about
    """
    errors = []
    warns = []

    # Backend validation - Qiskit or QuTiP REQUIRED
    valid_backends = ["qiskit", "qutip"]
    if config.backend not in valid_backends:
        errors.append(
            f"Invalid backend '{config.backend}'. Must be 'qiskit' or 'qutip'. "
            f"Install with: pip install qiskit qiskit-aer (or pip install qutip)"
        )

    # Verify backend library is actually installed
    if config.backend == "qiskit":
        try:
            import qiskit
            import qiskit_aer
        except ImportError:
            errors.append(
                "Backend 'qiskit' selected but Qiskit not installed. "
                "Install with: pip install qiskit qiskit-aer"
            )
    elif config.backend == "qutip":
        try:
            import qutip
        except ImportError:
            errors.append(
                "Backend 'qutip' selected but QuTiP not installed. "
                "Install with: pip install qutip"
            )

    # Basic parameter ranges
    if config.num_pairs <= 0:
        errors.append(f"num_pairs must be > 0, got {config.num_pairs}")

    if config.seed < 0:
        errors.append(f"seed must be >= 0, got {config.seed}")

    # Security parameters
    if not (0 <= config.pe_fraction <= 1):
        errors.append(f"pe_fraction must be in [0,1], got {config.pe_fraction}")

    if not (0 <= config.chsh_fraction <= 1):
        errors.append(f"chsh_fraction must be in [0,1], got {config.chsh_fraction}")

    if config.pe_fraction + config.chsh_fraction > 1:
        errors.append(
            f"pe_fraction ({config.pe_fraction}) + chsh_fraction ({config.chsh_fraction}) "
            f"cannot exceed 1.0"
        )

    if config.f_EC < 1.0:
        errors.append(f"f_EC (error correction inefficiency) must be >= 1.0, got {config.f_EC}")

    # Visibility / noise parameters
    if config.enable_visibility_reduction:
        if not (0 <= config.visibility <= 1):
            errors.append(f"visibility must be in [0,1], got {config.visibility}")

    if config.enable_depolarizing_noise:
        if not (0 <= config.depolarizing_alpha <= 1):
            errors.append(f"depolarizing_alpha must be in [0,1], got {config.depolarizing_alpha}")

    if config.enable_intrinsic_error:
        if not (0 <= config.intrinsic_error_ed <= 0.5):
            errors.append(f"intrinsic_error_ed must be in [0,0.5], got {config.intrinsic_error_ed}")

    # Loss parameters
    if config.enable_fiber_loss:
        if config.distance_km_A < 0:
            errors.append(f"distance_km_A must be >= 0, got {config.distance_km_A}")
        if config.distance_km_B < 0:
            errors.append(f"distance_km_B must be >= 0, got {config.distance_km_B}")
        if config.fiber_loss_dB_per_km < 0:
            errors.append(f"fiber_loss_dB_per_km must be >= 0, got {config.fiber_loss_dB_per_km}")

    if config.loss_dB_A < 0:
        errors.append(f"loss_dB_A must be >= 0, got {config.loss_dB_A}")
    if config.loss_dB_B < 0:
        errors.append(f"loss_dB_B must be >= 0, got {config.loss_dB_B}")

    # Detector parameters
    if config.enable_detector_loss:
        if not (0 <= config.heralding_efficiency <= 1):
            errors.append(f"heralding_efficiency must be in [0,1], got {config.heralding_efficiency}")
        if not (0 <= config.end_detector_efficiency <= 1):
            errors.append(f"end_detector_efficiency must be in [0,1], got {config.end_detector_efficiency}")

    # Dark counts
    if config.enable_dark_counts:
        if config.use_dark_cps:
            if config.dark_cps < 0:
                errors.append(f"dark_cps must be >= 0, got {config.dark_cps}")
        else:
            if not (0 <= config.dark_prob <= 1):
                errors.append(f"dark_prob must be in [0,1], got {config.dark_prob}")

    # Background
    if config.enable_background:
        if hasattr(config, 'background_cps') and config.background_cps < 0:
            errors.append(f"background_cps must be >= 0, got {config.background_cps}")

    # Timing parameters
    if config.enable_time_tagging:
        if config.repetition_rate_Hz <= 0:
            errors.append(f"repetition_rate_Hz must be > 0, got {config.repetition_rate_Hz}")
        if hasattr(config, 'gate_window_ns') and config.gate_window_ns <= 0:
            errors.append(f"gate_window_ns must be > 0, got {config.gate_window_ns}")
        if config.coincidence_window_ns <= 0:
            errors.append(f"coincidence_window_ns must be > 0, got {config.coincidence_window_ns}")

    # Timing jitter
    if config.enable_timing_jitter:
        if config.jitter_A_ns < 0:
            errors.append(f"jitter_A_ns must be >= 0, got {config.jitter_A_ns}")
        if config.jitter_B_ns < 0:
            errors.append(f"jitter_B_ns must be >= 0, got {config.jitter_B_ns}")

    # Deadtime
    if config.enable_deadtime:
        if config.deadtime_ns < 0:
            errors.append(f"deadtime_ns must be >= 0, got {config.deadtime_ns}")
        if hasattr(config, 'deadtime_A_ns') and config.deadtime_A_ns < 0:
            errors.append(f"deadtime_A_ns must be >= 0, got {config.deadtime_A_ns}")
        if hasattr(config, 'deadtime_B_ns') and config.deadtime_B_ns < 0:
            errors.append(f"deadtime_B_ns must be >= 0, got {config.deadtime_B_ns}")

    # Multi-pair SPDC (approximate; modeled via pair-number sampling + double-click squashing)
    if config.enable_multi_pair:
        warns.append(
            "enable_multi_pair=True uses an approximate SPDC multi-pair model (pair-number sampling) "
            "combined with double-click squashing. Results are approximate and intended for sensitivity studies."
        )

        # Validate parameters
        if config.spdc_brightness_mu < 0:
            errors.append(f"spdc_brightness_mu must be >= 0, got {config.spdc_brightness_mu}")

        # Validate SPDC distribution type
        valid_spdc_distributions = ["thermal", "poisson"]
        if hasattr(config, "spdc_distribution"):
            if config.spdc_distribution not in valid_spdc_distributions:
                errors.append(
                    f"Invalid spdc_distribution '{config.spdc_distribution}'. "
                    f"Must be one of {valid_spdc_distributions}"
                )

            # Warn if using Poisson (only appropriate for many-mode collection)
            if config.spdc_distribution == "poisson":
                warns.append(
                    "Using Poisson distribution for SPDC. This is typically appropriate for many-mode SPDC. "
                    "For single-mode SPDC, use 'thermal' (geometric) statistics."
                )

    # Saturation
    if config.enable_saturation:
        if config.saturation_rate <= 0:
            errors.append(f"saturation_rate must be > 0, got {config.saturation_rate}")

    # Double-click squashing
    if config.enable_double_click_squashing:
        valid_squashing_models = ["random", "discard"]
        if hasattr(config, 'squashing_model'):
            if config.squashing_model not in valid_squashing_models:
                errors.append(
                    f"Invalid squashing_model '{config.squashing_model}'. "
                    f"Must be one of {valid_squashing_models}"
                )

    # Conflicting configurations
    if config.enable_fiber_loss and config.enable_satellite:
        if config.distance_km_A > 0 or config.distance_km_B > 0:
            errors.append(
                "Cannot have both fiber distance and satellite enabled. "
                "Choose one or the other."
            )

    # Afterpulsing validation
    if config.enable_afterpulsing:
        if config.afterpulsing_prob < 0 or config.afterpulsing_prob > 1:
            errors.append(f"afterpulsing_prob must be in [0,1], got {config.afterpulsing_prob}")
        if hasattr(config, 'afterpulsing_tau_ns') and config.afterpulsing_tau_ns <= 0:
            errors.append(f"afterpulsing_tau_ns must be > 0, got {config.afterpulsing_tau_ns}")
        # Require time tagging for afterpulsing
        if not config.enable_time_tagging:
            errors.append(
                "Afterpulsing requires time tagging to be enabled. "
                "Set enable_time_tagging=True when using enable_afterpulsing=True."
            )

    # Bell state validation
    valid_bell_states = ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]
    if config.bell_state not in valid_bell_states:
        errors.append(f"Invalid bell_state '{config.bell_state}'. Must be one of {valid_bell_states}")

    # Warning for non-Phi+ Bell states (analytic backend supports all 4 properly)
    if config.bell_state != "phi_plus":
        warns.append(
            f"Bell state '{config.bell_state}': The analytic backend implements all Bell states "
            f"via density matrices. The legacy correlation backend only validates 'phi_plus'."
        )

    return errors, warns


def validate_and_raise(config: ExperimentConfig):
    """
    Validate config and raise ValueError if invalid.

    Prints warnings to console but only raises on errors.

    Args:
        config: ExperimentConfig to validate

    Raises:
        ValueError: If config has fatal errors
    """
    errors, warns = validate_config(config)

    # Print warnings
    if warns:
        for w in warns:
            warnings.warn(w, UserWarning)

    # Raise on errors
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)


__all__ = ['validate_config', 'validate_and_raise']
