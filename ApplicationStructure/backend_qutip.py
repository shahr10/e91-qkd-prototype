"""
QuTiP-based quantum backend for E91 QKD simulation.

This module implements a proper QuTiP backend that uses density matrices
and Born rule sampling for measurement outcomes.

Author: E91 QKD Simulation Team
Date: 2025-12-29
"""

from __future__ import annotations  # CRITICAL: Allows forward references without QuTiP installed

import numpy as np
from typing import Tuple, TYPE_CHECKING
import warnings

# Use TYPE_CHECKING to avoid runtime import errors
if TYPE_CHECKING:
    import qutip as qt

try:
    import qutip as qt
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False
    qt = None  # Define as None to prevent NameError
    warnings.warn("QuTiP not available. Install with: pip install qutip")


class QuTiPBackend:
    """
    QuTiP-based backend for E91 protocol measurements.

    Uses proper density matrix formalism with Born rule sampling.
    """

    def __init__(self, bell_state: str = "phi_plus", visibility: float = 1.0,
                 depolarizing_alpha: float = 0.0):
        """
        Initialize QuTiP backend with Bell state and noise parameters.

        Args:
            bell_state: Which Bell state to prepare
            visibility: Visibility parameter (Werner noise)
            depolarizing_alpha: Depolarizing noise parameter
        """
        if not QUTIP_AVAILABLE:
            raise ImportError("QuTiP backend requires qutip. Install with: pip install qutip")

        self.bell_state_name = bell_state
        self.visibility = visibility
        self.depolarizing_alpha = depolarizing_alpha

        # Create Bell state
        self.rho = self._create_bell_state(bell_state)

        # Apply noise channels
        if visibility < 1.0:
            self.rho = self._apply_werner_noise(self.rho, visibility)
        if depolarizing_alpha > 0:
            self.rho = self._apply_depolarizing_noise(self.rho, depolarizing_alpha)

    def _create_bell_state(self, bell_state: str) -> qt.Qobj:
        """
        Create Bell state density matrix using QuTiP.

        Args:
            bell_state: Name of Bell state

        Returns:
            Density matrix as QuTiP Qobj
        """
        # Computational basis states
        zero = qt.basis(2, 0)
        one = qt.basis(2, 1)

        # Create Bell states
        if bell_state == "phi_plus":
            # |Φ+⟩ = (|00⟩ + |11⟩)/√2
            psi = (qt.tensor(zero, zero) + qt.tensor(one, one)).unit()
        elif bell_state == "phi_minus":
            # |Φ-⟩ = (|00⟩ - |11⟩)/√2
            psi = (qt.tensor(zero, zero) - qt.tensor(one, one)).unit()
        elif bell_state == "psi_plus":
            # |Ψ+⟩ = (|01⟩ + |10⟩)/√2
            psi = (qt.tensor(zero, one) + qt.tensor(one, zero)).unit()
        elif bell_state == "psi_minus":
            # |Ψ-⟩ = (|01⟩ - |10⟩)/√2
            psi = (qt.tensor(zero, one) - qt.tensor(one, zero)).unit()
        else:
            raise ValueError(f"Unknown Bell state: {bell_state}")

        # Convert to density matrix
        return psi * psi.dag()

    def _apply_werner_noise(self, rho: qt.Qobj, visibility: float) -> qt.Qobj:
        """
        Apply Werner noise: ρ_out = V·ρ + (1-V)·I/4

        Args:
            rho: Input density matrix
            visibility: Visibility parameter

        Returns:
            Noisy density matrix
        """
        identity = qt.tensor(qt.qeye(2), qt.qeye(2)) / 4.0
        return visibility * rho + (1.0 - visibility) * identity

    def _apply_depolarizing_noise(self, rho: qt.Qobj, alpha: float) -> qt.Qobj:
        """
        Apply depolarizing noise via Werner state mixing.

        Implements: ρ_out = (1-α)·ρ + α·(I/4)

        This is mathematically equivalent to global depolarization and is
        trace-preserving by construction.

        Args:
            rho: Input density matrix
            alpha: Depolarizing parameter (0 = no noise, 1 = fully mixed)

        Returns:
            Noisy density matrix (properly normalized)
        """
        # Werner state mixing: ρ_out = (1-α)·ρ + α·(I/4)
        # where I/4 is the maximally mixed state
        I = qt.qeye(4)  # 4x4 identity for 2-qubit system
        maximally_mixed = I / 4.0

        rho_out = (1.0 - alpha) * rho + alpha * maximally_mixed

        # Verify trace preservation (should always be 1.0)
        trace = rho_out.tr()
        if abs(trace - 1.0) > 1e-10:
            # Renormalize if needed (shouldn't happen with correct implementation)
            rho_out = rho_out / trace

        return rho_out

    def _rotation_operator(self, angle: float) -> qt.Qobj:
        """
        Create rotation operator for polarization measurement.

        Args:
            angle: Measurement angle in radians

        Returns:
            Rotation operator as QuTiP Qobj
        """
        # R(θ) = cos(θ)|0⟩⟨0| + sin(θ)|0⟩⟨1| + sin(θ)|1⟩⟨0| - cos(θ)|1⟩⟨1|
        # This is equivalent to measuring in rotated basis
        return qt.Qobj([[np.cos(angle), np.sin(angle)],
                        [np.sin(angle), -np.cos(angle)]])

    def compute_measurement_probabilities(self, alice_angle: float, bob_angle: float
                                         ) -> Tuple[float, float, float, float]:
        """
        Compute joint measurement probabilities using Born rule.

        Args:
            alice_angle: Alice's measurement angle (radians)
            bob_angle: Bob's measurement angle (radians)

        Returns:
            Tuple of (P(00), P(01), P(10), P(11))
        """
        # Measurement operators (projectors in rotated basis)
        # For Alice
        R_a = self._rotation_operator(alice_angle)
        P_a0 = qt.tensor(R_a * qt.basis(2, 0) * (R_a * qt.basis(2, 0)).dag(), qt.qeye(2))
        P_a1 = qt.tensor(R_a * qt.basis(2, 1) * (R_a * qt.basis(2, 1)).dag(), qt.qeye(2))

        # For Bob
        R_b = self._rotation_operator(bob_angle)
        P_b0 = qt.tensor(qt.qeye(2), R_b * qt.basis(2, 0) * (R_b * qt.basis(2, 0)).dag())
        P_b1 = qt.tensor(qt.qeye(2), R_b * qt.basis(2, 1) * (R_b * qt.basis(2, 1)).dag())

        # Compute probabilities via Born rule: P(a,b) = Tr[P_a ⊗ P_b ρ]
        p00 = np.real((P_a0 * P_b0 * self.rho).tr())
        p01 = np.real((P_a0 * P_b1 * self.rho).tr())
        p10 = np.real((P_a1 * P_b0 * self.rho).tr())
        p11 = np.real((P_a1 * P_b1 * self.rho).tr())

        # Normalize to ensure probabilities sum to 1
        total = p00 + p01 + p10 + p11
        if total > 0:
            p00 /= total
            p01 /= total
            p10 /= total
            p11 /= total

        return (p00, p01, p10, p11)

    def sample_measurements(self, alice_angles: np.ndarray, bob_angles: np.ndarray,
                           rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample measurement outcomes from quantum state.

        Args:
            alice_angles: Alice's measurement angles
            bob_angles: Bob's measurement angles
            rng: Random number generator

        Returns:
            Tuple of (alice_results, bob_results) as binary arrays
        """
        n = len(alice_angles)
        alice_results = np.zeros(n, dtype=int)
        bob_results = np.zeros(n, dtype=int)

        for i in range(n):
            # Compute probabilities for this measurement setting
            p00, p01, p10, p11 = self.compute_measurement_probabilities(
                alice_angles[i], bob_angles[i]
            )

            # Sample outcome
            outcome = rng.choice(4, p=[p00, p01, p10, p11])

            if outcome == 0:  # 00
                alice_results[i] = 0
                bob_results[i] = 0
            elif outcome == 1:  # 01
                alice_results[i] = 0
                bob_results[i] = 1
            elif outcome == 2:  # 10
                alice_results[i] = 1
                bob_results[i] = 0
            else:  # 11
                alice_results[i] = 1
                bob_results[i] = 1

        return alice_results, bob_results


def create_qutip_backend(bell_state: str, visibility: float = 1.0,
                         depolarizing_alpha: float = 0.0) -> QuTiPBackend:
    """
    Factory function to create QuTiP backend.

    Args:
        bell_state: Bell state name
        visibility: Visibility (Werner noise)
        depolarizing_alpha: Depolarizing noise

    Returns:
        QuTiPBackend instance
    """
    return QuTiPBackend(bell_state, visibility, depolarizing_alpha)


__all__ = ['QuTiPBackend', 'create_qutip_backend', 'QUTIP_AVAILABLE']
