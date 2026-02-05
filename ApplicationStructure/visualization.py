"""
================================================================================
VISUALIZATION FOR E91 QKD
================================================================================

Comprehensive results visualization functions.

Author: Tyler Barr
Version: 7.0.0 Modular
Date: 2025

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from .models import ExperimentConfig, ExperimentResults

# Import constants - with fallbacks if not available
try:
    from .config import QuantumConstants
    CHSH_CLASSICAL_BOUND = QuantumConstants.CHSH_CLASSICAL_BOUND
    CHSH_QUANTUM_MAX = QuantumConstants.CHSH_QUANTUM_MAX
    QBER_THRESHOLD = QuantumConstants.QBER_THRESHOLD
except ImportError:
    CHSH_CLASSICAL_BOUND = 2.0
    CHSH_QUANTUM_MAX = 2.0 * np.sqrt(2)
    QBER_THRESHOLD = 0.11


# ============================================================================
# RESULTS VISUALIZATION
# ============================================================================

def create_results_plots(config: ExperimentConfig, results: ExperimentResults):
    """
    Create results visualization with clean, consistent labels and legends.

    Args:
        config: Experiment configuration
        results: Experiment results

    Returns:
        Matplotlib figure with 9 subplots showing comprehensive results
    """
    fig = plt.figure(figsize=(20, 14), constrained_layout=False)
    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.4, top=0.95, bottom=0.05, left=0.05, right=0.95)

    # CHSH Parameter
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(
        ['Measured', 'Classical\nBound', 'Quantum\nMax'],
        [results.chsh_S, CHSH_CLASSICAL_BOUND, CHSH_QUANTUM_MAX],
        color=['#3b82f6', '#ef4444', '#22c55e'], alpha=0.8, edgecolor='black', linewidth=1.5,
    )
    ax1.set_ylabel('CHSH S', fontsize=11, fontweight='bold')
    ax1.set_title('Bell Test Result', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=CHSH_CLASSICAL_BOUND, color='red', linestyle='--', linewidth=2, label='Classical bound (2.0)')
    ax1.legend(fontsize=9, loc='upper right')

    # QBER
    ax2 = fig.add_subplot(gs[0, 1])
    qber_percent = results.qber * 100
    threshold_percent = QBER_THRESHOLD * 100
    q_color = '#10b981' if qber_percent < threshold_percent else '#f59e0b'
    ax2.bar(['QBER'], [qber_percent], color=q_color, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=threshold_percent, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax2.set_ylabel('QBER (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Quantum Bit Error Rate', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(axis='y', alpha=0.3)

    # Efficiency Pipeline
    ax3 = fig.add_subplot(gs[0, 2])
    stages = ['Generated', 'Detected', 'Sifted', 'Final Key']
    counts = [
        results.num_pairs_generated,
        results.num_pairs_detected,
        results.num_sifted,
        results.num_key_bits,
    ]
    colors_pipeline = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444']
    ax3.bar(stages, counts, color=colors_pipeline, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Number of Bits', fontsize=11, fontweight='bold')
    ax3.set_title('Key Generation Pipeline', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.tick_params(axis='x', rotation=15)
    for i, count in enumerate(counts):
        pct = (count / max(results.num_pairs_generated, 1)) * 100
        ax3.text(i, count, f'{pct:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Secret Fractions
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(
        ['Asymptotic', 'Finite-Key'],
        [results.secret_fraction_asymptotic, results.secret_fraction_finite],
        color=['#93c5fd', '#60a5fa'], alpha=0.9, edgecolor='black', linewidth=1.2,
    )
    ax4.set_ylabel('Secret Fraction', fontsize=11, fontweight='bold')
    ax4.set_title('Secret Key Fractions', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 1)
    ax4.grid(axis='y', alpha=0.3)

    # Key Rates
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(
        ['Asymptotic', 'Finite-Key'],
        [results.key_rate_asymptotic, results.key_rate_finite],
        color=['#fda4af', '#fb7185'], alpha=0.9, edgecolor='black', linewidth=1.2,
    )
    ax5.set_ylabel('Key Rate (bits/pair)', fontsize=11, fontweight='bold')
    ax5.set_title('Key Rates', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)

    # Correlators Heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    corr_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            corr_matrix[i, j] = results.correlators.get(f'E_{i}{j}', 0)
    im = ax6.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
    ax6.set_xticks([0, 1, 2])
    ax6.set_yticks([0, 1, 2])
    ax6.set_xticklabels(['Bob 0', 'Bob 1', 'Bob 2'])
    ax6.set_yticklabels(['Alice 0', 'Alice 1', 'Alice 2'])
    ax6.set_title('Measurement Correlators', fontsize=12, fontweight='bold')
    for i in range(3):
        for j in range(3):
            ax6.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center', color='black', fontsize=9, fontweight='bold')
    plt.colorbar(im, ax=ax6, label='Correlation E(a_i, b_j)', fraction=0.046)

    # Measurement Angles
    ax7 = fig.add_subplot(gs[2, 0], projection='polar')
    for angle in config.alice_angles:
        ax7.plot([angle, angle], [0, 0.8], 'b-', linewidth=3, alpha=0.8)
        ax7.plot(angle, 0.8, 'bo', markersize=10)
    for angle in config.bob_angles:
        ax7.plot([angle, angle], [0, 0.6], 'r--', linewidth=3, alpha=0.8)
        ax7.plot(angle, 0.6, 'rs', markersize=10)
    ax7.set_title('Measurement Angles', fontsize=12, fontweight='bold', pad=20)
    ax7.legend(['Alice', 'Bob'], loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

    # Summary Statistics (ASCII-only text to avoid encoding issues)
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.axis('off')
    summary_text = (
        "EXPERIMENT SUMMARY\n\n"
        f"Generated: {results.num_pairs_generated:,}\n"
        f"Detected: {results.num_pairs_detected:,}\n"
        f"Detection eta: {results.detection_efficiency*100:.2f}%\n\n"
        f"Sifted: {results.num_sifted:,}\n"
        f"Sifting eta: {results.sifting_efficiency*100:.2f}%\n\n"
        f"QBER: {results.qber*100:.3f}%\n"
        f"CHSH S: {results.chsh_S:.4f}\n"
        f"Bell: {'YES' if results.chsh_S > CHSH_CLASSICAL_BOUND else 'NO'}\n\n"
        f"Final Key: {results.num_key_bits:,}\n"
        f"Rate (F): {results.key_rate_finite:.6f}\n\n"
        f"Runtime: {results.execution_time:.2f}s\n"
    )
    ax8.text(
        0.1,
        0.95,
        summary_text,
        transform=ax8.transAxes,
        fontsize=9,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )

    # Security card
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    is_secure = (results.chsh_S > CHSH_CLASSICAL_BOUND) and (results.qber < QBER_THRESHOLD)
    status = 'SECURE' if is_secure else 'INSECURE'
    quality_text = (
        "SECURITY ASSESSMENT\n\n"
        f"Status: {status}\n\n"
        f"Bell Test: {'PASS' if results.chsh_S > CHSH_CLASSICAL_BOUND else 'FAIL'}\n"
        f"QBER Test: {'PASS' if results.qber < QBER_THRESHOLD else 'FAIL'}\n\n"
        f"Key Rate: {results.key_rate_finite:.6f}\n"
        f"Efficiency: {results.detection_efficiency*100:.1f}%\n\n"
        f"Preset: {config.preset_name}\n"
        f"Backend: {config.backend}\n"
    )
    color = 'lightgreen' if is_secure else 'lightcoral'
    ax9.text(
        0.1,
        0.95,
        quality_text,
        transform=ax9.transAxes,
        fontsize=9,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor=color, alpha=0.5),
    )

    plt.suptitle(
        f'E91 QKD Experiment Results - {config.preset_name}', fontsize=14, fontweight='bold', y=0.98
    )
    return fig


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'create_results_plots',
]
