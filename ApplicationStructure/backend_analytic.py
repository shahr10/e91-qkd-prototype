"""
Pure NumPy analytic backend for E91 QKD simulation.

This backend uses density matrix formalism and Born rule to compute
quantum measurement outcomes without requiring QuTiP or Qiskit.

Implementation:
- Builds density matrices for Bell states
- Applies Werner mixing for visibility < 1
- Computes Born probabilities for measurement outcomes
- Samples from multinomial distribution

Author: E91 QKD Simulation Team
Date: 2025-12-29
"""

import numpy as np
from typing import Tuple


# Pauli matrices (for constructing measurement operators)
PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def build_bell_state_density_matrix(bell_state: str) -> np.ndarray:
    """
    Build density matrix for a pure Bell state.

    Bell states in computational basis:
    - phi_plus:  |Φ+⟩ = (|00⟩ + |11⟩)/√2
    - phi_minus: |Φ-⟩ = (|00⟩ - |11⟩)/√2
    - psi_plus:  |Ψ+⟩ = (|01⟩ + |10⟩)/√2
    - psi_minus: |Ψ-⟩ = (|01⟩ - |10⟩)/√2

    Args:
        bell_state: One of "phi_plus", "phi_minus", "psi_plus", "psi_minus"

    Returns:
        4×4 density matrix ρ = |ψ⟩⟨ψ|
    """
    # Define Bell state vectors (4-dimensional, basis |00⟩, |01⟩, |10⟩, |11⟩)
    if bell_state == "phi_plus":
        psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)  # (|00⟩ + |11⟩)/√2
    elif bell_state == "phi_minus":
        psi = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)  # (|00⟩ - |11⟩)/√2
    elif bell_state == "psi_plus":
        psi = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)  # (|01⟩ + |10⟩)/√2
    elif bell_state == "psi_minus":
        psi = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)  # (|01⟩ - |10⟩)/√2
    else:
        raise ValueError(f"Unknown Bell state: {bell_state}")

    # ρ = |ψ⟩⟨ψ|
    rho = np.outer(psi, psi.conj())
    return rho


def apply_werner_mixing(rho_pure: np.ndarray, visibility: float) -> np.ndarray:
    """
    Apply Werner state mixing to a pure Bell state.

    Werner state: ρ = V·ρ_pure + (1-V)·I/4

    Args:
        rho_pure: Pure Bell state density matrix (4×4)
        visibility: Visibility parameter V ∈ [0,1]

    Returns:
        Mixed state density matrix
    """
    identity = np.eye(4, dtype=complex)
    rho_mixed = visibility * rho_pure + (1 - visibility) * identity / 4
    return rho_mixed


def measurement_operator_polarization(theta: float, outcome: int) -> np.ndarray:
    """
    Construct polarization measurement operator for a single qubit.

    Polarization measurement at angle θ projects onto:
    - outcome=0: |θ⟩ = cos(θ)|H⟩ + sin(θ)|V⟩
    - outcome=1: |θ⊥⟩ = -sin(θ)|H⟩ + cos(θ)|V⟩

    Args:
        theta: Measurement angle in radians
        outcome: 0 or 1

    Returns:
        2×2 projection operator M = |ψ⟩⟨ψ|
    """
    if outcome == 0:
        # |θ⟩ = cos(θ)|H⟩ + sin(θ)|V⟩
        psi = np.array([np.cos(theta), np.sin(theta)], dtype=complex)
    else:
        # |θ⊥⟩ = -sin(θ)|H⟩ + cos(θ)|V⟩
        psi = np.array([-np.sin(theta), np.cos(theta)], dtype=complex)

    M = np.outer(psi, psi.conj())
    return M


def compute_born_probabilities(
    rho: np.ndarray,
    theta_A: float,
    theta_B: float
) -> np.ndarray:
    """
    Compute Born rule probabilities for all four outcomes.

    P(a,b|θA,θB) = Tr[ρ · (M_a^A ⊗ M_b^B)]

    Args:
        rho: 4×4 density matrix
        theta_A: Alice's measurement angle (radians)
        theta_B: Bob's measurement angle (radians)

    Returns:
        Array [P(0,0), P(0,1), P(1,0), P(1,1)]
    """
    probs = np.zeros(4)

    for a in [0, 1]:
        for b in [0, 1]:
            # Build measurement operators
            M_A = measurement_operator_polarization(theta_A, a)
            M_B = measurement_operator_polarization(theta_B, b)

            # Tensor product M_A ⊗ M_B
            M_AB = np.kron(M_A, M_B)

            # Born rule: P(a,b) = Tr[ρ · M_AB]
            prob = np.trace(rho @ M_AB).real

            # Store in order: (0,0), (0,1), (1,0), (1,1)
            idx = 2 * a + b
            probs[idx] = prob

    # Normalize (handle numerical errors)
    probs = np.clip(probs, 0, 1)
    probs /= probs.sum()

    return probs


class AnalyticBackend:
    """
    Pure NumPy analytic backend for quantum state sampling.

    Uses density matrix formalism to compute Born rule probabilities
    and sample measurement outcomes.
    """

    def __init__(self, bell_state: str, visibility: float, rng: np.random.Generator):
        """
        Initialize analytic backend.

        Args:
            bell_state: One of "phi_plus", "phi_minus", "psi_plus", "psi_minus"
            visibility: Visibility parameter V ∈ [0,1]
            rng: NumPy random number generator
        """
        self.bell_state = bell_state
        self.visibility = visibility
        self.rng = rng

        # Build density matrix
        rho_pure = build_bell_state_density_matrix(bell_state)
        self.rho = apply_werner_mixing(rho_pure, visibility)

    def sample(
        self,
        angles_A: np.ndarray,
        angles_B: np.ndarray,
        n: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample measurement outcomes for n pairs.

        Args:
            angles_A: Alice's measurement angles (radians), shape (n,)
            angles_B: Bob's measurement angles (radians), shape (n,)
            n: Number of pairs to sample

        Returns:
            (alice_results, bob_results): Binary arrays of shape (n,)
        """
        alice_results = np.zeros(n, dtype=int)
        bob_results = np.zeros(n, dtype=int)

        # For efficiency, group by unique angle pairs
        angle_pairs = list(zip(angles_A, angles_B))
        unique_pairs = list(set(angle_pairs))

        for theta_A, theta_B in unique_pairs:
            # Find all instances of this angle pair
            mask = (angles_A == theta_A) & (angles_B == theta_B)
            count = np.sum(mask)

            if count == 0:
                continue

            # Compute Born probabilities for this angle pair
            probs = compute_born_probabilities(self.rho, theta_A, theta_B)

            # Sample from multinomial distribution
            # outcomes: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            outcomes = self.rng.choice(4, size=count, p=probs)

            # Decode outcomes into Alice and Bob results
            alice_outcomes = outcomes // 2  # 0 or 1
            bob_outcomes = outcomes % 2      # 0 or 1

            # Store results
            alice_results[mask] = alice_outcomes
            bob_results[mask] = bob_outcomes

        return alice_results, bob_results


def create_analytic_backend(
    bell_state: str,
    visibility: float,
    rng: np.random.Generator
) -> AnalyticBackend:
    """
    Factory function for creating analytic backend.

    Args:
        bell_state: One of "phi_plus", "phi_minus", "psi_plus", "psi_minus"
        visibility: Visibility parameter V ∈ [0,1]
        rng: NumPy random number generator

    Returns:
        AnalyticBackend instance
    """
    return AnalyticBackend(bell_state, visibility, rng)


# Validation: Compute correlation function for Phi+ state
def validate_phi_plus_correlation():
    """
    Validate that Phi+ state gives E(a,b) = cos(2(a-b)) for polarization.

    This is a self-test function to verify the backend implementation.
    """
    print("Validating Phi+ correlation function...")

    rho_pure = build_bell_state_density_matrix("phi_plus")
    rho = apply_werner_mixing(rho_pure, visibility=1.0)

    # Test angles
    test_cases = [
        (0, 0, 1.0),           # E(0,0) = 1
        (0, np.pi/4, 0.0),     # E(0,π/4) = 0
        (0, np.pi/2, -1.0),    # E(0,π/2) = -1
        (np.pi/8, 3*np.pi/8, 0.0),  # E(π/8, 3π/8) = 0
    ]

    for theta_A, theta_B, expected_E in test_cases:
        probs = compute_born_probabilities(rho, theta_A, theta_B)

        # Correlation: E = P(0,0) + P(1,1) - P(0,1) - P(1,0)
        E = probs[0] + probs[3] - probs[1] - probs[2]

        # Expected: E = cos(2(θA - θB))
        E_theory = np.cos(2 * (theta_A - theta_B))

        error = abs(E - E_theory)
        status = "PASS" if error < 1e-10 else "FAIL"

        print(f"  thetaA={theta_A:.3f}, thetaB={theta_B:.3f}: "
              f"E={E:.6f}, theory={E_theory:.6f}, error={error:.2e} {status}")

    print("Validation complete.\n")


if __name__ == "__main__":
    # Run validation when module is executed directly
    validate_phi_plus_correlation()
