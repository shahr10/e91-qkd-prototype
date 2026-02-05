"""
================================================================================
PARAMETRIC SWEEP ANALYSIS FOR E91 QKD
================================================================================

Functions for running parametric sweeps and generating analysis plots.

Author: Tyler Barr
Version: 7.0.0 Modular
Date: 2025

================================================================================
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Callable
from .models import ExperimentConfig, ExperimentResults
from .quantum_protocol import E91Protocol


# ============================================================================
# PARAMETRIC SWEEP
# ============================================================================

def run_parameter_sweep(
    base_config: ExperimentConfig,
    param_name: str,
    param_values: List[float],
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> List[Tuple[float, ExperimentResults]]:
    """
    Run parametric sweep over a range of parameter values.

    Args:
        base_config: Base experiment configuration
        param_name: Name of parameter to sweep
        param_values: List of values for the parameter
        progress_callback: Optional callback function(progress, message)

    Returns:
        List of tuples (parameter_value, ExperimentResults)
    """
    results = []
    for i, val in enumerate(param_values):
        config = copy.deepcopy(base_config)

        # Handle special case for distance_km (needs to set both A and B)
        if param_name == 'distance_km':
            config.distance_km_A = val
            config.distance_km_B = val
        else:
            # Direct attribute setting for all other parameters
            if hasattr(config, param_name):
                setattr(config, param_name, val)
            else:
                raise ValueError(f"Parameter '{param_name}' not found in ExperimentConfig")

        if progress_callback:
            progress_callback((i + 1) / len(param_values),
                            f"Sweep {i+1}/{len(param_values)}: {param_name}={val:.4f}")

        protocol = E91Protocol(config)
        result = protocol.run()
        results.append((val, result))

    return results


def create_sweep_plots(
    sweep_results: List[Tuple[float, ExperimentResults]],
    param_name: str,
    param_label: Optional[str] = None
) -> dict:
    """
    Create sweep plots from parametric sweep results.

    Args:
        sweep_results: List of (parameter_value, ExperimentResults) tuples
        param_name: Name of parameter that was swept
        param_label: Human-readable parameter label (optional)

    Returns:
        Dictionary of matplotlib figures {plot_name: figure}
    """
    if param_label is None:
        param_label = param_name.replace('_', ' ').title()

    param_vals = [val for val, _ in sweep_results]
    qber_vals = [r.qber for _, r in sweep_results]
    chsh_vals = [r.chsh_S for _, r in sweep_results]
    key_rate_asymp = [r.key_rate_asymptotic for _, r in sweep_results]
    key_rate_finite = [r.key_rate_finite for _, r in sweep_results]
    detection_eff = [r.detection_efficiency for _, r in sweep_results]

    plots = {}

    # Plot 1: S vs QBER
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    scatter = ax1.scatter(qber_vals, chsh_vals, c=param_vals, cmap='viridis',
                         s=100, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='Classical Bound')
    ax1.axhline(y=2*np.sqrt(2), color='green', linestyle='--', linewidth=2, label='Quantum Max')
    ax1.axvline(x=0.11, color='orange', linestyle='--', linewidth=2, label='QBER Threshold')
    ax1.set_xlabel('QBER', fontsize=12, fontweight='bold')
    ax1.set_ylabel('CHSH S', fontsize=12, fontweight='bold')
    ax1.set_title('CHSH S vs QBER', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label=param_label)
    plt.tight_layout()
    plots['s_vs_qber'] = fig1

    # Plot 2: Key Rate vs QBER
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(qber_vals, key_rate_asymp, 'bo-', linewidth=2, label='Asymptotic', alpha=0.7)
    ax2.plot(qber_vals, key_rate_finite, 'rs-', linewidth=2, label='Finite-Key', alpha=0.7)
    ax2.axvline(x=0.11, color='orange', linestyle='--', linewidth=2, label='QBER Threshold')
    ax2.set_xlabel('QBER', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Key Rate', fontsize=12, fontweight='bold')
    ax2.set_title('Key Rate vs QBER', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    plots['keyrate_vs_qber'] = fig2

    # Plot 3: Key Rate vs Parameter
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(param_vals, key_rate_asymp, 'bo-', linewidth=2, label='Asymptotic', alpha=0.7)
    ax3.plot(param_vals, key_rate_finite, 'rs-', linewidth=2, label='Finite-Key', alpha=0.7)
    ax3.set_xlabel(param_label, fontsize=12, fontweight='bold')
    ax3.set_ylabel('Key Rate', fontsize=12, fontweight='bold')
    ax3.set_title(f'Key Rate vs {param_label}', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    plt.tight_layout()
    plots['keyrate_vs_param'] = fig3

    # Plot 4: CHSH vs Parameter
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.plot(param_vals, chsh_vals, 'go-', linewidth=2, alpha=0.7)
    ax4.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='Classical Bound')
    ax4.set_xlabel(param_label, fontsize=12, fontweight='bold')
    ax4.set_ylabel('CHSH S', fontsize=12, fontweight='bold')
    ax4.set_title(f'CHSH S vs {param_label}', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    plt.tight_layout()
    plots['chsh_vs_param'] = fig4

    # Plot 5: Detection Efficiency
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    ax5.plot(param_vals, detection_eff, 'mo-', linewidth=2, alpha=0.7)
    ax5.set_xlabel(param_label, fontsize=12, fontweight='bold')
    ax5.set_ylabel('Detection Efficiency', fontsize=12, fontweight='bold')
    ax5.set_title(f'Detection Efficiency vs {param_label}', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    plots['detection_vs_param'] = fig5

    return plots


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'run_parameter_sweep',
    'create_sweep_plots',
]
