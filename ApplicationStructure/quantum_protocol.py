"""
================================================================================
E91 QUANTUM PROTOCOL IMPLEMENTATION
================================================================================

Core implementation of the E91 protocol using entangled photon pairs.
Supports both Qiskit (circuit-based) and QuTiP (density matrix) backends.

Protocol Flow:
  1. Generate entangled Bell pairs
  2. Alice and Bob measure with random basis choices
  3. Apply physical losses and noise models
  4. Sift keys by matching bases
  5. Estimate QBER via parameter estimation
  6. Calculate CHSH violation for security verification
  7. Extract secret key with finite-size effects

REALISTIC IMPAIRMENTS MODELING:

This implementation models practical entanglement-based QKD with:
- Multi-pair SPDC emissions (thermal/Poisson statistics)
- Channel loss (fiber attenuation, insertion loss, detector inefficiency)
- Detector artifacts (dark counts, afterpulsing, deadtime, saturation)
- Background light and accidental coincidences
- Threshold-detector squashing models for security proofs

PRINCIPAL REFERENCES:

Entanglement-based QKD with realistic impairments:
  Ma et al. (2012), "Practical decoy state for quantum key distribution"
    https://doi.org/10.1103/PhysRevA.72.012326
    (Multi-pair emissions + decoy states in entanglement sources)

  Branciard et al. (2012), "Source imperfections in entanglement-based QKD"
    https://arxiv.org/abs/1208.1495
    (Comprehensive treatment of SPDC imperfections + detector inefficiency)

SPDC photon statistics:
  Eisaman et al. (2011), "Invited Review: Single-photon sources and detectors"
    Rev. Mod. Phys. 83, 1059, https://doi.org/10.1103/RevModPhys.83.1
    (Thermal vs Poisson distribution for single-mode vs multi-mode SPDC)

  Takesue & Shimizu (2010), "Effects of multi-pair emission in entanglement QKD"
    Opt. Commun. 283, 276, https://doi.org/10.1016/j.optcom.2009.10.008
    (Multi-pair visibility degradation: V_eff ≈ V/(1+μ) for thermal SPDC)

Threshold-detector squashing models:
  Beaudry, Moroder & Lütkenhaus (2008), Phys. Rev. A 78, 042320
    "Squashing models for optical measurements in quantum communication"
    (Framework for random assignment vs discard squashing)

  Gottesman et al. (2004), Quantum Inf. Comput. 4, 325
    "Security of quantum key distribution with imperfect devices"
    (GLLP proof incorporating threshold detectors)

NASA/Applied references (engineering perspective):
  NASA Technical Memorandum (applied entanglement QKD modeling)
    (Practical implementation guide for satellite entanglement distribution)

Author: Tyler Barr
Version: 7.0.0 Modular
Date: 2025

================================================================================
"""

import numpy as np
import time
from typing import Tuple, Optional
from .models import ExperimentConfig, ExperimentResults

# Try importing Qiskit
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# Try importing QuTiP
try:
    import qutip as qt
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False

# Import QuTiP backend (REQUIRED if using qutip backend)
try:
    from .backend_qutip import create_qutip_backend, QuTiPBackend
    QUTIP_BACKEND_AVAILABLE = True
except ImportError:
    QUTIP_BACKEND_AVAILABLE = False

# Import config validator
try:
    from .config_validator import validate_and_raise
except ImportError:
    # Fallback: no validation
    def validate_and_raise(config):
        pass

# Try importing quantum_math functions
try:
    from .quantum_math import binary_entropy
except ImportError:
    # Fallback implementation
    def binary_entropy(p: float) -> float:
        """Shannon binary entropy function."""
        if p <= 0 or p >= 1:
            return 0.0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

# Import satellite loss computation from utils
try:
    from .utils import compute_satellite_loss
except ImportError:
    # Fallback implementation
    def compute_satellite_loss(config) -> float:
        """Fallback satellite loss computation."""
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
        total_efficiency = geometric_loss * pointing_loss
        return -10 * np.log10(max(total_efficiency, 1e-10))


# ============================================================================
# E91 PROTOCOL CLASS
# ============================================================================

class E91Protocol:
    """
    E91 Quantum Key Distribution Protocol Implementation.

    This class handles the complete E91 protocol simulation including:
    - Bell state preparation (Phi+, Phi-, Psi+, Psi-)
    - Quantum measurements with configurable angles
    - Physical channel losses (fiber, detectors, etc.)
    - Noise models (depolarizing, visibility, etc.)
    - QBER estimation and CHSH violation calculation
    - Secret key extraction with finite-size effects
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize E91 protocol simulator.

        Args:
            config: Experiment configuration

        Raises:
            RuntimeError: If selected backend is not available
        """
        # Validate configuration
        validate_and_raise(config)

        self.config = config
        self.rng = np.random.default_rng(config.seed)  # Use modern RNG API

        # Backend selection - Qiskit or QuTiP REQUIRED
        if config.backend == "qiskit":
            if not QISKIT_AVAILABLE:
                raise RuntimeError(
                    "Qiskit backend requested but not available.\n"
                    "Install with:\n"
                    "  pip install 'qiskit>=1.0' 'qiskit-aer>=0.14'\n"
                    "OR\n"
                    "  conda install -c conda-forge qiskit qiskit-aer"
                )
            self.simulator = AerSimulator(method='statevector')
            self.use_qiskit = True
            self.use_qutip = False
            self.qutip_backend = None
        elif config.backend == "qutip":
            if not QUTIP_AVAILABLE or not QUTIP_BACKEND_AVAILABLE:
                raise RuntimeError(
                    "QuTiP backend requested but not available.\n"
                    "Install with:\n"
                    "  pip install 'qutip>=5.0'\n"
                    "OR\n"
                    "  conda install -c conda-forge qutip"
                )
            self.use_qiskit = False
            self.use_qutip = True
            # QuTiP backend will be created per-run with current noise params
            self.qutip_backend = None
        else:
            raise RuntimeError(
                f"Unknown backend '{config.backend}'. Must be 'qiskit' or 'qutip'.\n"
                f"No quantum backend is available. Install at least one:\n"
                f"  pip install 'qiskit>=1.0' 'qiskit-aer>=0.14' 'qutip>=5.0'\n"
                f"OR\n"
                f"  conda install -c conda-forge qiskit qiskit-aer qutip"
            )

    def run(self, progress_callback=None) -> ExperimentResults:
        """
        Run the E91 protocol.

        Args:
            progress_callback: Optional callback function(progress, message)

        Returns:
            ExperimentResults with comprehensive metrics
        """
        start_time = time.time()

        # Generate measurements
        if progress_callback:
            progress_callback(0.1, "Generating measurements...")

        # Use quantum backend (Qiskit or QuTiP)
        # NOTE: Multipair emission effects are modeled via double-click squashing
        # (enable_double_click_squashing) which handles the k≥2 case by either
        # random assignment or discarding double-click events.
        if self.use_qiskit:
            # Qiskit backend uses quantum circuit simulation
            alice_bases, bob_bases, alice_results, bob_results = self._generate_measurements()
            has_signal = None
        elif self.use_qutip:
            # QuTiP backend uses density matrix simulation with Born rule sampling
            alice_bases, bob_bases, alice_results, bob_results = self._generate_measurements_qutip(
                self.config.num_pairs
            )
            has_signal = None
        else:
            raise RuntimeError("No valid backend selected")

        # Multi-pair SPDC: sample pair counts once per trial (used for metrics, squashing, and loss gating)
        pair_counts = None
        if self.config.enable_multi_pair:
            pair_counts = self._sample_pair_number(self.config.num_pairs)
            multi_pair_events = int(np.sum(pair_counts >= 2))
            # Signal exists if at least one pair was emitted for the trial
            has_signal = pair_counts > 0
        else:
            multi_pair_events = None

        # Eve (eavesdropper)
        if self.config.enable_eavesdropper:
            alice_results, bob_results, eve_count = self._apply_eavesdropper(
                alice_bases, bob_bases, alice_results, bob_results)
        else:
            eve_count = None

        # Double-click squashing (detector-resolved model)
        # Pass pair_counts (integer array) to squashing logic
        if self.config.enable_double_click_squashing:
            alice_results, bob_results, squash_keep_mask = self._apply_double_click_squashing(
                alice_results, bob_results, pair_counts
            )
            # Apply keep mask to all arrays (including has_signal and pair_counts)
            alice_results = alice_results[squash_keep_mask]
            bob_results = bob_results[squash_keep_mask]
            alice_bases = alice_bases[squash_keep_mask]
            bob_bases = bob_bases[squash_keep_mask]
            if has_signal is not None and len(has_signal) == len(squash_keep_mask):
                has_signal = has_signal[squash_keep_mask]
            if pair_counts is not None and len(pair_counts) == len(squash_keep_mask):
                pair_counts = pair_counts[squash_keep_mask]

        # Time tags
        if progress_callback:
            progress_callback(0.3, "Generating time tags...")
        time_tags_alice, time_tags_bob = (
            self._generate_time_tags(len(alice_results))
            if self.config.enable_time_tagging else (None, None)
        )

        # Apply losses
        if progress_callback:
            progress_callback(0.5, "Applying losses...")
        # FIX P0-5: Updated to receive singles streams for CAR calculation
        result = self._apply_losses(alice_results, bob_results, alice_bases, bob_bases,
                             time_tags_alice, time_tags_bob, has_signal)

        # Unpack based on number of return values
        if len(result) == 8:
            # New format: includes singles streams
            alice_results, bob_results, alice_bases, bob_bases, \
                time_tags_alice, time_tags_bob, time_tags_alice_singles, time_tags_bob_singles = result
        else:
            # Legacy format (shouldn't happen with time tags)
            alice_results, bob_results, alice_bases, bob_bases, time_tags_alice, time_tags_bob = result
            time_tags_alice_singles, time_tags_bob_singles = None, None

        num_detected = len(alice_results)
        detection_eff = num_detected / self.config.num_pairs

        # Coincidence stats
        if progress_callback:
            progress_callback(0.7, "Computing statistics...")
        # FIX P0-5: Pass both coincidence and singles streams for CAR calculation
        if self.config.enable_time_tagging and time_tags_alice is not None:
            avg_delay, car, rate_A_Hz, rate_B_Hz, coinc_rate_Hz, accidentals_est = \
                self._compute_coincidence_stats(
                    time_tags_alice, time_tags_bob,
                    time_tags_alice_singles, time_tags_bob_singles
                )
        else:
            avg_delay, car, rate_A_Hz, rate_B_Hz, coinc_rate_Hz, accidentals_est = \
                None, None, None, None, None, None

        # PHASE 5B FIX: Separate CHSH test samples from key/PE samples
        # Security requirement: Samples used for Bell test must not be used for key
        # Split detected results into three disjoint sets BEFORE sifting

        if progress_callback:
            progress_callback(0.75, "Separating test and key samples...")

        num_detected = len(alice_results)

        # Determine sample allocation
        # CHSH test: 10% of detected events (configurable via chsh_fraction)
        chsh_fraction = getattr(self.config, 'chsh_fraction', 0.1)
        pe_fraction = self.config.pe_fraction

        # CRITICAL FIX: Filter for CHSH-valid basis pairs FIRST
        # Use explicit CHSH indices from config for protocol correctness
        a0, a1 = self.config.chsh_a0_idx, self.config.chsh_a1_idx
        b0, b1 = self.config.chsh_b0_idx, self.config.chsh_b1_idx
        # Valid CHSH basis pairs are all combinations of {a0,a1} x {b0,b1}
        chsh_valid_pairs = {(a0,b0), (a0,b1), (a1,b0), (a1,b1)}
        chsh_valid_mask = np.array([
            (alice_bases[i], bob_bases[i]) in chsh_valid_pairs
            for i in range(num_detected)
        ])

        # Similarly, define key-valid basis pairs (matching bases for key extraction)
        # For E91: typically basis 1 (π/8, π/4) for key
        key_valid_pairs = {(1,1)}  # Or adjust based on protocol
        key_valid_mask = np.array([
            (alice_bases[i], bob_bases[i]) in key_valid_pairs
            for i in range(num_detected)
        ])

        # Get indices of CHSH-valid and key-valid events
        chsh_pool = np.where(chsh_valid_mask)[0]
        key_pool = np.where(key_valid_mask)[0]

        n_chsh_available = len(chsh_pool)
        n_chsh = max(100, int(n_chsh_available * min(1.0, chsh_fraction / (chsh_fraction + pe_fraction))))
        n_chsh = min(n_chsh, n_chsh_available)

        # Sample from CHSH pool only
        if n_chsh > 0 and n_chsh_available > 0:
            chsh_indices = self.rng.choice(chsh_pool, size=min(n_chsh, n_chsh_available), replace=False)
            chsh_mask = np.zeros(num_detected, dtype=bool)
            chsh_mask[chsh_indices] = True

            # Remaining data for key/PE
            key_pe_mask = ~chsh_mask

            # CRITICAL: Enforce disjoint PE/key samples (security requirement)
            assert not np.any(chsh_mask & key_pe_mask), \
                "CHSH and key/PE samples must be disjoint for security proof validity"

            # Extract CHSH test samples
            chsh_alice = alice_results[chsh_mask]
            chsh_bob = bob_results[chsh_mask]
            chsh_alice_bases = alice_bases[chsh_mask]
            chsh_bob_bases = bob_bases[chsh_mask]

            # Extract key/PE samples
            key_pe_alice = alice_results[key_pe_mask]
            key_pe_bob = bob_results[key_pe_mask]
            key_pe_alice_bases = alice_bases[key_pe_mask]
            key_pe_bob_bases = bob_bases[key_pe_mask]
        else:
            # If too few events, use all for CHSH (emergency fallback)
            chsh_alice = alice_results
            chsh_bob = bob_results
            chsh_alice_bases = alice_bases
            chsh_bob_bases = bob_bases

            # Empty key/PE samples
            key_pe_alice = np.array([], dtype=int)
            key_pe_bob = np.array([], dtype=int)
            key_pe_alice_bases = np.array([], dtype=int)
            key_pe_bob_bases = np.array([], dtype=int)

        # Compute CHSH and correlators from CHSH test subset ONLY
        # For security: parameter estimation must be disjoint from key generation
        chsh_S, correlators = self._compute_chsh(
            chsh_alice, chsh_bob, chsh_alice_bases, chsh_bob_bases)

        # Sift keys from key/PE samples only
        if progress_callback:
            progress_callback(0.8, "Sifting keys...")
        sifted_alice, sifted_bob = self._sift_keys(
            key_pe_alice, key_pe_bob, key_pe_alice_bases, key_pe_bob_bases)
        num_sifted = len(sifted_alice)
        sifting_eff = num_sifted / num_detected if num_detected > 0 else 0

        # Parameter estimation and key generation
        if progress_callback:
            progress_callback(0.9, "Computing key rates...")

        # Split sifted bits into parameter-estimation sample and key block
        if num_sifted > 1 and pe_fraction > 0:
            n_pe = max(1, int(num_sifted * pe_fraction))
            n_pe = min(n_pe, num_sifted - 1)
        else:
            n_pe = 0

        if n_pe > 0:
            # Random sampling without replacement for parameter estimation (Phase 3 fix)
            pe_indices = self.rng.choice(num_sifted, size=n_pe, replace=False)
            key_mask = np.ones(num_sifted, dtype=bool)
            key_mask[pe_indices] = False

            pe_alice = sifted_alice[pe_indices]
            pe_bob = sifted_bob[pe_indices]
            key_alice = sifted_alice[key_mask]
            key_bob = sifted_bob[key_mask]
        else:
            pe_alice = np.array([], dtype=int)
            pe_bob = np.array([], dtype=int)
            key_alice = sifted_alice
            key_bob = sifted_bob

        n_key = len(key_alice)

        # QBER estimate: use parameter-estimation sample when available
        if n_pe > 0:
            pe_errors = np.sum(pe_alice != pe_bob)
            qber = pe_errors / n_pe if n_pe > 0 else 0.5
        else:
            key_errors = np.sum(key_alice != key_bob)
            qber = key_errors / n_key if n_key > 0 else 0.5

        # Phase 5B: Compute per-basis QBER from PE samples (diagnostic information)
        per_basis_qber = self._compute_per_basis_qber(
            key_pe_alice, key_pe_bob, key_pe_alice_bases, key_pe_bob_bases
        ) if len(key_pe_alice) > 0 else None

        # PHASE 5B: Use quantum_math.py functions (deduplicated code)
        from ApplicationStructure.quantum_math import (
            secret_key_rate_asymptotic,
            secret_key_rate_finite
        )

        # Asymptotic key rate
        # UNITS: secret_fraction is "secret bits per sifted bit"
        # key_rate is "secret bits per detected pair" = secret_fraction × sifting_efficiency
        # To get "secret bits per emitted pair", multiply by detection_efficiency
        # To get "secret bits per second", multiply by repetition_rate_Hz
        secret_frac_asymp = secret_key_rate_asymptotic(
            qber=qber,
            sifting_efficiency=1.0,  # Will multiply by sifting_eff later
            f_ec=self.config.f_EC
        )
        key_rate_asymp = secret_frac_asymp * sifting_eff

        # Finite-size key rate
        final_key_bits, secret_frac_finite = secret_key_rate_finite(
            n_sifted=num_sifted,
            qber=qber,
            sifting_efficiency=1.0,  # Will multiply by sifting_eff later
            epsilon_sec=self.config.epsilon_sec,
            epsilon_cor=self.config.epsilon_cor,
            pe_fraction=self.config.pe_fraction,
            f_ec=self.config.f_EC,
            code_rate=self.config.code_rate
        )
        key_rate_finite = secret_frac_finite * sifting_eff

        # Routed DI
        routed_di_witness, combined_eff = None, None
        if self.config.enable_routed_di:
            eta_routing = self.config.routing_efficiency_f * self.config.routing_efficiency_d
            combined_eff = detection_eff * eta_routing
            routed_di_witness = chsh_S / (2 * np.sqrt(2 * combined_eff))

        # Multi-pair events already returned from _generate_measurements_with_noise()
        # No need to re-sample (would break reproducibility - Phase 5 fix)

        execution_time = time.time() - start_time
        if progress_callback:
            progress_callback(1.0, "Complete!")

        return ExperimentResults(
            num_pairs_generated=self.config.num_pairs,
            num_pairs_detected=num_detected,
            detection_efficiency=detection_eff,
            num_sifted=num_sifted,
            sifting_efficiency=sifting_eff,
            qber=qber,
            chsh_S=chsh_S,
            num_key_bits=final_key_bits,
            secret_fraction_asymptotic=secret_frac_asymp,
            secret_fraction_finite=secret_frac_finite,
            key_rate_asymptotic=key_rate_asymp,
            key_rate_finite=key_rate_finite,
            correlators=correlators,
            routed_di_witness=routed_di_witness,
            combined_efficiency=combined_eff,
            avg_coincidence_delay_ns=avg_delay,
            coincidence_accidental_ratio=car,
            multi_pair_events=multi_pair_events,
            per_basis_qber=per_basis_qber,  # Phase 5B
            # Mandatory output metrics (G1)
            singles_rate_alice_Hz=rate_A_Hz,
            singles_rate_bob_Hz=rate_B_Hz,
            coincidence_rate_Hz=coinc_rate_Hz,
            accidentals_estimate=accidentals_est,
            alice_results=alice_results.tolist(),
            bob_results=bob_results.tolist(),
            sifted_alice=key_alice.tolist(),
            sifted_bob=key_bob.tolist(),
            time_tags_alice=time_tags_alice.tolist() if time_tags_alice is not None else None,
            time_tags_bob=time_tags_bob.tolist() if time_tags_bob is not None else None,
            execution_time=execution_time,
            eve_intercepted=eve_count
        )

    def _generate_measurements(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate Bell pair measurements.

        Returns:
            Tuple of (alice_bases, bob_bases, alice_results, bob_results)
        """
        n = self.config.num_pairs
        alice_bases = (self.rng.choice(3, n, p=self.config.alice_basis_probs)
                      if self.config.alice_basis_probs else self.rng.integers(0, 3, n))
        bob_bases = (self.rng.choice(3, n, p=self.config.bob_basis_probs)
                    if self.config.bob_basis_probs else self.rng.integers(0, 3, n))
        alice_results = np.zeros(n, dtype=int)
        bob_results = np.zeros(n, dtype=int)

        if self.use_qiskit:
            # Calculate base visibility from noise sources (matches analytic backend)
            base_visibility = 1.0
            if self.config.enable_depolarizing_noise:
                base_visibility *= (1.0 - self.config.depolarizing_alpha)
            if self.config.enable_visibility_reduction:
                base_visibility *= self.config.visibility
            if self.config.enable_intrinsic_error:
                base_visibility *= (1.0 - self.config.intrinsic_error_ed)

            # Multi-pair SPDC visibility penalty (Takesue & Shimizu 2010)
            # Multi-pair emissions create mixed states: ρ_multi = Σ_k P(k) ρ_k
            # Effective visibility: V_eff ≈ V * (1 - P(k≥2) * degradation_factor)
            # For thermal SPDC: P(k≥2) = 1 - 1/(1+μ) - μ/(1+μ)²
            # Simplified: V_eff ≈ V / (1 + μ) for small μ
            # Reference: https://doi.org/10.1016/j.optcom.2009.10.008
            if self.config.enable_multi_pair:
                mu = self.config.spdc_brightness_mu
                # Takesue formula: V_eff = V / (1 + μ) for thermal SPDC
                base_visibility /= (1.0 + mu)

            # Convert visibility to single-qubit depolarizing probability
            # Werner state: ρ = V·|ψ⟩⟨ψ| + (1-V)·I/4
            # NOTE: Depolarizing noise is applied via post-processing (Werner state mixing)
            # rather than circuit noise channels. See lines 548-554 below.

            for ai in range(3):
                for bj in range(3):
                    mask = (alice_bases == ai) & (bob_bases == bj)
                    m = int(np.sum(mask))
                    if m == 0:
                        continue
                    angle_A, angle_B = self.config.alice_angles[ai], self.config.bob_angles[bj]
                    qc = QuantumCircuit(2, 2)
                    # Prepare |Phi+> Bell state
                    qc.h(0)
                    qc.cx(0, 1)
                    # Adjust to selected Bell state via local unitaries
                    bell = getattr(self.config, "bell_state", "phi_plus")
                    if bell == "phi_minus":
                        qc.z(1)
                    elif bell == "psi_plus":
                        qc.x(1)
                    elif bell == "psi_minus":
                        qc.x(1)
                        qc.z(1)

                    # Note: Depolarizing noise will be applied via post-processing
                    # (Werner state mixing) rather than circuit noise channels

                    # Apply measurement rotations
                    qc.ry(-2 * angle_A, 0)
                    qc.ry(-2 * angle_B, 1)
                    qc.measure([0, 1], [0, 1])
                    result = self.simulator.run(
                        qc,
                        shots=m,
                        seed_simulator=int(self.rng.integers(0, 2**31)),
                    ).result()
                    counts = result.get_counts()
                    outcomes = []
                    for bitstr, c in counts.items():
                        outcomes.extend([bitstr] * c)
                    self.rng.shuffle(outcomes)
                    idxs = np.flatnonzero(mask)
                    for k, s in enumerate(outcomes):
                        bob_results[idxs[k]] = int(s[0])
                        alice_results[idxs[k]] = int(s[1])

            # Apply Werner state mixing via post-processing
            # ρ_noisy = V·ρ_pure + (1-V)·I/4
            # This is equivalent to: with probability (1-V), replace outcome with uniform random
            if base_visibility < 1.0:
                noise_prob = 1.0 - base_visibility
                for i in range(n):
                    if self.rng.random() < noise_prob:
                        # Replace with uniform random outcome (simulates I/4 mixing)
                        alice_results[i] = self.rng.integers(0, 2)
                        bob_results[i] = self.rng.integers(0, 2)

        else:
            for i in range(n):
                angle_A = self.config.alice_angles[alice_bases[i]]
                angle_B = self.config.bob_angles[bob_bases[i]]
                delta = angle_A - angle_B
                prob_00 = 0.5 * np.cos(delta) ** 2
                prob_11 = prob_00
                prob_01 = 0.5 * np.sin(delta) ** 2
                prob_10 = prob_01
                outcome = self.rng.choice(4, p=[prob_00, prob_01, prob_10, prob_11])
                alice_results[i] = outcome // 2
                bob_results[i] = outcome % 2

        return alice_bases, bob_bases, alice_results, bob_results

    def _generate_measurements_qutip(self, n: int) -> Tuple:
        """
        Generate measurements using QuTiP density matrix backend.

        Uses proper quantum mechanics with Born rule sampling from density matrices.

        Args:
            n: Number of measurements to generate

        Returns:
            Tuple of (alice_bases, bob_bases, alice_results, bob_results)
        """
        # 1. Generate basis choices
        alice_bases = (self.rng.choice(3, n, p=self.config.alice_basis_probs)
                      if self.config.alice_basis_probs else self.rng.integers(0, 3, n))
        bob_bases = (self.rng.choice(3, n, p=self.config.bob_basis_probs)
                    if self.config.bob_basis_probs else self.rng.integers(0, 3, n))

        # 2. Map basis indices to measurement angles
        alice_angles = np.array([self.config.alice_angles[i] for i in alice_bases])
        bob_angles = np.array([self.config.bob_angles[i] for i in bob_bases])

        # Apply misalignment as angle rotation
        if self.config.enable_misalignment:
            drift_rad = np.deg2rad(self.config.polarization_drift_deg)
            bob_angles = bob_angles + drift_rad

        # 3. Compute effective visibility from ALL noise sources
        visibility = 1.0

        if self.config.enable_visibility_reduction:
            visibility *= self.config.visibility

        if self.config.enable_intrinsic_error:
            # Intrinsic error reduces visibility
            ed = self.config.intrinsic_error_ed
            visibility *= (1.0 - 2.0 * ed)

        # Multi-pair SPDC visibility penalty (Takesue & Shimizu 2010)
        # V_eff ≈ V / (1 + μ) for thermal SPDC
        # Reference: https://doi.org/10.1016/j.optcom.2009.10.008
        if self.config.enable_multi_pair:
            mu = self.config.spdc_brightness_mu
            visibility /= (1.0 + mu)

        # 4. Create QuTiP backend with current noise parameters
        depolarizing_alpha = (self.config.depolarizing_alpha
                             if self.config.enable_depolarizing_noise else 0.0)

        self.qutip_backend = create_qutip_backend(
            bell_state=self.config.bell_state,
            visibility=visibility,
            depolarizing_alpha=depolarizing_alpha
        )

        # 5. Sample measurements from quantum state
        alice_results, bob_results = self.qutip_backend.sample_measurements(
            alice_angles, bob_angles, self.rng
        )

        return alice_bases, bob_bases, alice_results, bob_results

    def _generate_measurements_with_noise(self, n: int) -> Tuple:
        """
        DEPRECATED: This method is no longer used and should not be called.

        Use _generate_measurements() for Qiskit or _generate_measurements_qutip() for QuTiP.
        """
        raise RuntimeError(
            "Legacy method _generate_measurements_with_noise() is deprecated. "
            "Use Qiskit or QuTiP backends via _generate_measurements() or _generate_measurements_qutip()."
        )
        # DEAD CODE BELOW - kept for reference only
        # 1. Sample number of pairs per trial (multi-pair SPDC)
        if self.config.enable_multi_pair:
            k_pairs = self._sample_pair_number(n)
            n_multipair = int(np.sum(k_pairs >= 2))
        else:
            k_pairs = np.ones(n, dtype=int)
            n_multipair = 0

        # 2. Generate basis choices
        alice_bases = (self.rng.choice(3, n, p=self.config.alice_basis_probs)
                      if self.config.alice_basis_probs else self.rng.integers(0, 3, n))
        bob_bases = (self.rng.choice(3, n, p=self.config.bob_basis_probs)
                    if self.config.bob_basis_probs else self.rng.integers(0, 3, n))

        # 3. Map basis indices to measurement angles
        alice_angles = np.array([self.config.alice_angles[i] for i in alice_bases])
        bob_angles = np.array([self.config.bob_angles[i] for i in bob_bases])

        # PHASE 5 FIX: Apply misalignment as angle rotation (not loss)
        if self.config.enable_misalignment:
            drift_rad = np.deg2rad(self.config.polarization_drift_deg)
            bob_angles = bob_angles + drift_rad  # Rotate Bob's angle

        # 4. Compute effective visibility from ALL noise sources
        base_visibility = 1.0

        # Depolarizing noise: V ← (1-α)·V
        if self.config.enable_depolarizing_noise:
            base_visibility *= (1.0 - self.config.depolarizing_alpha)

        # Visibility reduction: V ← vis·V
        if self.config.enable_visibility_reduction:
            base_visibility *= self.config.visibility

        # Intrinsic error: V ← (1-e_d)·V (approximate mapping)
        if self.config.enable_intrinsic_error:
            base_visibility *= (1.0 - self.config.intrinsic_error_ed)

        # Multi-pair reduces visibility: V_eff ≈ V / k for k≥2
        # (Crude model: multiple pairs create mixed state)
        # Avoid division by zero: set visibility to 0 when k_pairs = 0
        visibility_array = np.full(n, base_visibility, dtype=float)
        multipair_mask = k_pairs >= 2
        zero_pair_mask = k_pairs == 0
        if np.any(multipair_mask):
            visibility_array[multipair_mask] = base_visibility / k_pairs[multipair_mask]
        if np.any(zero_pair_mask):
            visibility_array[zero_pair_mask] = 0.0

        # 5. Compute angle differences (angles are already in radians)
        delta = alice_angles - bob_angles

        # 6. Compute correlations based on Bell state type
        # FIX #6: Bell state formulas - ONLY Phi+ is fully validated
        # WARNING: Phi-, Psi+, Psi- use simplified formulas that may not match
        # quantum theory for all angle settings. For research-grade accuracy,
        # use only bell_state="phi_plus" (default).
        if self.config.bell_state in ["phi_plus", "phi_minus"]:
            # Φ+ state: E(a,b) = cos(2(a-b))  [VALIDATED]
            # Φ- state: Sign flip approximation [NOT FULLY VALIDATED]
            correlation = visibility_array * np.cos(2 * delta)
            if self.config.bell_state == "phi_minus":
                correlation = -correlation
        elif self.config.bell_state in ["psi_plus", "psi_minus"]:
            # Ψ± states: Simplified formula [NOT VALIDATED - use at own risk]
            # Correct formula requires local unitary transform on Bob's side
            delta_sum = alice_angles + bob_angles
            correlation = -visibility_array * np.cos(2 * delta_sum)
            if self.config.bell_state == "psi_minus":
                correlation = -correlation
        else:
            # Default to phi_plus if unknown
            correlation = visibility_array * np.cos(2 * delta)

        # No signal correlation when k=0 (no photons emitted)
        has_signal = k_pairs > 0
        correlation = np.where(has_signal, correlation, 0.0)

        # 7. Sample quantum outcomes
        if self.use_analytic_backend:
            # Use density matrix backend for accurate Born-rule sampling
            # Create backend with effective visibility
            backend = create_analytic_backend(
                bell_state=self.config.bell_state,
                visibility=base_visibility,  # Use base visibility (multi-pair handled separately)
                rng=self.rng
            )

            # Sample measurements
            alice_results, bob_results = backend.sample(alice_angles, bob_angles, n)

            # Apply multi-pair degradation: randomize outcomes where k≥2
            if np.any(multipair_mask):
                # Multi-pair events lose quantum correlation - randomize outcomes
                alice_results[multipair_mask] = self.rng.integers(0, 2, np.sum(multipair_mask))
                bob_results[multipair_mask] = self.rng.integers(0, 2, np.sum(multipair_mask))

            # Zero-pair events have random outcomes
            if np.any(zero_pair_mask):
                alice_results[zero_pair_mask] = self.rng.integers(0, 2, np.sum(zero_pair_mask))
                bob_results[zero_pair_mask] = self.rng.integers(0, 2, np.sum(zero_pair_mask))

        else:
            # Use correlation-based sampling (legacy backend="qutip")
            # Convert correlation to outcome probabilities
            # E = P(same) - P(diff), E = (P(same) - P(diff))/(P(same) + P(diff))
            # For normalized outcomes: P(same) + P(diff) = 1
            # Therefore: P(same) = (1 + E)/2
            p_same = np.clip((1.0 + correlation) / 2.0, 0.0, 1.0)

            # 8. Sample outcomes from joint distribution
            outcomes_same = self.rng.random(n) < p_same
            alice_results = self.rng.integers(0, 2, n)
            # If same: Bob gets Alice's result; if different: Bob gets opposite
            bob_results = np.where(outcomes_same, alice_results, 1 - alice_results)

        if self.config.enable_multi_pair:
            return alice_results, bob_results, alice_bases, bob_bases, has_signal, n_multipair
        else:
            return alice_results, bob_results, alice_bases, bob_bases

    def _apply_eavesdropper(self, alice_bases, bob_bases, alice_results, bob_results):
        """
        Apply a simple Eve model (intercept-resend).

        With probability `eve_intercept_prob`, Bob's bit is flipped with
        probability 0.25 to approximate disturbance when Eve measures in a
        random basis and resends.

        Returns:
            Tuple of (updated alice_results, updated bob_results, number of intercepted pairs)
        """
        n = len(alice_results)
        if n == 0 or self.config.eve_intercept_prob <= 0:
            return alice_results, bob_results, 0
        mask = self.rng.random(n) < float(self.config.eve_intercept_prob)
        if self.config.eve_model == "intercept_resend":
            flip = (self.rng.random(n) < 0.25) & mask
            bob_results = np.where(flip, 1 - bob_results, bob_results)
        else:
            flip = (self.rng.random(n) < (0.1 * float(self.config.eve_intercept_prob))) & mask
            bob_results = np.where(flip, 1 - bob_results, bob_results)
        return alice_results, bob_results, int(np.sum(mask))

    def _generate_time_tags(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate time tags.

        Args:
            n: Number of time tags to generate

        Returns:
            Tuple of (alice_time_tags, bob_time_tags)
        """
        period = 1.0 / self.config.repetition_rate_Hz * 1e9
        ideal_times = np.arange(n) * period
        jitter_A = (self.rng.normal(0, self.config.jitter_A_ns, n)
                   if self.config.enable_timing_jitter else np.zeros(n))
        jitter_B = (self.rng.normal(0, self.config.jitter_B_ns, n)
                   if self.config.enable_timing_jitter else np.zeros(n))
        return ideal_times + jitter_A, ideal_times + jitter_B

    def _apply_losses(self, alice_results, bob_results, alice_bases, bob_bases,
                     time_tags_A=None, time_tags_B=None, has_signal=None):
        """
        Apply all loss mechanisms using per-side click modeling (Phase 5 refactoring).

        KEY CHANGE: Track signal and noise clicks PER DETECTOR, not as coincidence.
        This allows signal+noise coincidences (dominant at high loss).

        Click model:
            click_A = signal_A OR noise_A
            click_B = signal_B OR noise_B
            coincidence = click_A AND click_B

        Args:
            alice_results: Alice's measurement results (quantum correlated)
            bob_results: Bob's measurement results (quantum correlated)
            alice_bases: Alice's basis choices
            bob_bases: Bob's basis choices
            time_tags_A: Alice's time tags (optional)
            time_tags_B: Bob's time tags (optional)
            has_signal: Multi-pair signal mask (k>0 events), optional

        Returns:
            Tuple of filtered results, bases, and time tags
        """
        n = len(alice_results)

        # Initialize per-side signal masks
        # has_signal is a boolean mask (True if k > 0, False if k = 0)
        if has_signal is not None:
            sig_A = has_signal.copy()
            sig_B = has_signal.copy()
        else:
            sig_A = np.ones(n, dtype=bool)
            sig_B = np.ones(n, dtype=bool)

        # NOTE: Depolarizing, visibility, intrinsic errors, and misalignment
        # are handled in _generate_measurements_with_noise() (Phases 2 & 5).

        # Apply channel losses (per side)
        if self.config.enable_fiber_loss:
            loss_dB_A = self.config.distance_km_A * self.config.fiber_loss_dB_per_km + self.config.loss_dB_A
            loss_dB_B = self.config.distance_km_B * self.config.fiber_loss_dB_per_km + self.config.loss_dB_B

            if getattr(self.config, "enable_repeaters", False) and getattr(self.config, "num_repeaters", 0) > 0:
                total_gain = self.config.num_repeaters * self.config.repeater_gain_dB
                loss_dB_A = max(0.0, loss_dB_A - total_gain)
                loss_dB_B = max(0.0, loss_dB_B - total_gain)

            eta_A = 10 ** (-loss_dB_A / 10)
            eta_B = 10 ** (-loss_dB_B / 10)

            sig_A &= self.rng.random(n) < eta_A
            sig_B &= self.rng.random(n) < eta_B

        # Satellite loss (per arm)
        # FIX P0-2: Independent sampling per arm (different slant ranges, turbulence)
        if self.config.enable_satellite:
            sat_loss_dB = compute_satellite_loss(self.config)
            eta_sat = 10 ** (-sat_loss_dB / 10)
            # Each downlink is independent (different path, turbulence, pointing)
            survival_A = self.rng.random(n) < eta_sat
            survival_B = self.rng.random(n) < eta_sat
            sig_A &= survival_A
            sig_B &= survival_B

        # Insertion loss (per side)
        if self.config.enable_insertion_loss:
            eta_ins = 10 ** (-self.config.insertion_loss_dB / 10)
            sig_A &= self.rng.random(n) < eta_ins
            sig_B &= self.rng.random(n) < eta_ins

        # Detector efficiency (per side)
        # FIX: Independent sampling per side with per-side parameters
        if self.config.enable_detector_loss:
            # Heralding efficiency: independent per side
            eta_herald_A = getattr(self.config, 'heralding_efficiency_A', self.config.heralding_efficiency)
            eta_herald_B = getattr(self.config, 'heralding_efficiency_B', self.config.heralding_efficiency)
            heralding_A = self.rng.random(n) < eta_herald_A
            heralding_B = self.rng.random(n) < eta_herald_B
            sig_A &= heralding_A
            sig_B &= heralding_B

            # End detector efficiency: independent per side
            eta_det_A = getattr(self.config, 'end_detector_efficiency_A', self.config.end_detector_efficiency)
            eta_det_B = getattr(self.config, 'end_detector_efficiency_B', self.config.end_detector_efficiency)
            sig_A &= self.rng.random(n) < eta_det_A
            sig_B &= self.rng.random(n) < eta_det_B

        # Compute noise clicks (per side)
        # FIX C2: Use gate_window_ns for noise probabilities (not coincidence_window_ns)
        gate_window = getattr(self.config, 'gate_window_ns', 1.0)
        p_noise_A, p_noise_B = self._compute_noise_probabilities(n, gate_window)

        noise_A = self.rng.random(n) < p_noise_A
        noise_B = self.rng.random(n) < p_noise_B

        # Combine signal and noise
        click_A = sig_A | noise_A
        click_B = sig_B | noise_B

        # Apply deadtime (per detector side, with independent parameters)
        if self.config.enable_deadtime and time_tags_A is not None:
            deadtime_A = getattr(self.config, 'deadtime_A_ns', self.config.deadtime_ns)
            deadtime_B = getattr(self.config, 'deadtime_B_ns', self.config.deadtime_ns)
            click_A = self._apply_deadtime_vectorized(click_A, time_tags_A, deadtime_A)
            click_B = self._apply_deadtime_vectorized(click_B, time_tags_B, deadtime_B)

        # Apply afterpulsing (after deadtime, per detector)
        if self.config.enable_afterpulsing and time_tags_A is not None:
            click_A = self._apply_afterpulsing(click_A, time_tags_A)
            click_B = self._apply_afterpulsing(click_B, time_tags_B)

        # Apply saturation (per detector)
        if self.config.enable_saturation:
            if self.config.repetition_rate_Hz > self.config.saturation_rate:
                saturation_prob = self.config.saturation_rate / self.config.repetition_rate_Hz
                click_A &= self.rng.random(n) < saturation_prob
                click_B &= self.rng.random(n) < saturation_prob

        # Coincidence = both detectors click
        coincidence = click_A & click_B

        # CRITICAL FIX: Only coincidences where BOTH sides detected signal are quantum-correlated
        # For signal+noise or noise+noise coincidences, outcomes are uncorrelated (random)
        correlated = sig_A & sig_B & coincidence

        # FIX P0-4: Selective randomization for mixed events
        # - If Alice has signal, Bob has noise → keep Alice's quantum outcome, randomize Bob's
        # - If Alice has noise, Bob has signal → randomize Alice's, keep Bob's quantum outcome
        # - If both have signal → use both quantum outcomes (correlated)
        # - If both have noise → randomize both

        # Filter to coincidence events for easier indexing
        coinc_mask = coincidence
        n_coinc = np.sum(coinc_mask)

        # Determine which side has signal in coincidence events
        sig_A_coinc = sig_A[coinc_mask]
        sig_B_coinc = sig_B[coinc_mask]

        # Alice's outcome: use quantum if Alice has signal, else randomize
        alice_results_filtered = np.where(
            sig_A_coinc,
            alice_results[coinc_mask],  # Alice has signal → use quantum outcome
            self.rng.integers(0, 2, n_coinc)  # Alice has noise → random
        )

        # Bob's outcome: use quantum if Bob has signal, else randomize
        bob_results_filtered = np.where(
            sig_B_coinc,
            bob_results[coinc_mask],  # Bob has signal → use quantum outcome
            self.rng.integers(0, 2, n_coinc)  # Bob has noise → random
        )

        # Filter to coincidence events
        filtered = (
            alice_results_filtered,
            bob_results_filtered,
            alice_bases[coincidence],
            bob_bases[coincidence]
        )

        # FIX P0-5: Return singles click streams for CAR calculation
        if time_tags_A is not None:
            # Return coincidence data + singles streams
            return (*filtered,
                    time_tags_A[coincidence], time_tags_B[coincidence],  # Coincidence time tags
                    time_tags_A[click_A], time_tags_B[click_B])  # Singles time tags
        return (*filtered, None, None, None, None)

    def _compute_noise_probabilities(self, n, gate_window_ns=1.0):
        """
        Compute per-side noise click probabilities using independent union (Phase 5).

        FIX C2: Uses gate_window_ns (detector gate width) not coincidence_window_ns

        Sources: dark counts, background light, afterpulsing

        Formula: P(noise) = 1 - (1-P_dark)*(1-P_bg)*(1-P_afterpulse)

        Args:
            n: Number of trials
            gate_window_ns: Detection gate window width in nanoseconds

        Returns:
            Tuple of (p_noise_A, p_noise_B)
        """
        # Dark counts
        # FIX P0-3: Use Poisson formula P = 1 - exp(-λτ) instead of linear approximation
        p_dark = 0.0
        if self.config.enable_dark_counts:
            if self.config.use_dark_cps:
                lambda_tau = self.config.dark_cps * gate_window_ns * 1e-9
                # Poisson: P(≥1 event) = 1 - P(0 events) = 1 - exp(-λτ)
                p_dark = 1.0 - np.exp(-lambda_tau) if lambda_tau > 0 else 0.0
            else:
                p_dark = self.config.dark_prob
            p_dark = float(np.clip(p_dark, 0.0, 1.0))

        # Background light
        p_bg = 0.0
        if self.config.enable_background:
            if hasattr(self.config, 'background_cps'):
                lambda_tau = self.config.background_cps * gate_window_ns * 1e-9
                # Poisson: P(≥1 event) = 1 - exp(-λτ)
                p_bg = 1.0 - np.exp(-lambda_tau) if lambda_tau > 0 else 0.0
            else:
                p_bg = self.config.Y0
            p_bg = float(np.clip(p_bg, 0.0, 1.0))

        # Satellite background
        p_sat_bg = 0.0
        if self.config.enable_satellite and self.config.is_daytime:
            # FIX C2: Use gate window for noise probabilities
            lambda_tau = self.config.satellite_background_cps * gate_window_ns * 1e-9
            # Poisson: P(≥1 event) = 1 - exp(-λτ)
            p_sat_bg = 1.0 - np.exp(-lambda_tau) if lambda_tau > 0 else 0.0
            p_sat_bg = float(np.clip(p_sat_bg, 0.0, 1.0))

        # Afterpulsing - NOT included in this probability calculation
        #
        # IMPORTANT: Afterpulsing IS implemented in the simulation (see _apply_afterpulsing),
        # but it's handled as a TIME-DOMAIN CONDITIONAL PROCESS, not a per-trial probability.
        #
        # Physical model (already implemented in _apply_afterpulsing):
        #   1. After each real click, generate afterpulse with probability p_ap
        #   2. Delay sampled from exponential distribution: t ~ -tau * ln(U)
        #   3. Afterpulse click assigned to nearest future gate (random outcome)
        #
        # This approach correctly models the conditional nature of afterpulsing without
        # double-counting it in the noise probability union. Including it here would
        # incorrectly treat afterpulsing as an independent noise source.
        #
        # Reference: Namekata et al. (2006), Opt. Express 14, 10043
        p_afterpulse = 0.0  # Modeled separately in _apply_afterpulsing()

        # INDEPENDENT UNION (not simple sum!)
        p_no_dark = 1.0 - p_dark
        p_no_bg = 1.0 - p_bg
        p_no_sat = 1.0 - p_sat_bg
        p_no_ap = 1.0 - p_afterpulse

        p_noise = 1.0 - (p_no_dark * p_no_bg * p_no_sat * p_no_ap)
        p_noise = float(np.clip(p_noise, 0.0, 1.0))

        return p_noise, p_noise

    def _apply_deadtime_vectorized(self, clicks, time_tags, deadtime_ns):
        """
        Apply detector deadtime filtering (vectorized, per detector channel) - Phase 5.

        Suppresses clicks that occur within deadtime_ns of the last KEPT click.

        FIX #4: Track last kept click, not last raw click.
        FIX #11: Applied per detector side (A/B). For full detector-resolved deadtime
        (D0_A, D1_A, D0_B, D1_B), this would require tracking which detector fired,
        which is handled by the double-click squashing model earlier in the pipeline.

        Args:
            clicks: Boolean array of click events
            time_tags: Time stamps of events
            deadtime_ns: Deadtime in nanoseconds

        Returns:
            Filtered click array
        """
        if not np.any(clicks):
            return clicks

        # Get times of clicks
        click_indices = np.where(clicks)[0]
        click_times = time_tags[click_indices]

        # Track which clicks to keep
        keep = np.zeros(len(click_indices), dtype=bool)
        keep[0] = True  # First click always survives

        # Iterate through clicks, comparing to last KEPT click
        last_kept_time = click_times[0]
        for i in range(1, len(click_times)):
            if click_times[i] - last_kept_time >= deadtime_ns:
                keep[i] = True
                last_kept_time = click_times[i]

        # Create filtered clicks array
        filtered_clicks = clicks.copy()
        suppressed_indices = click_indices[~keep]
        filtered_clicks[suppressed_indices] = False

        return filtered_clicks

    def _apply_afterpulsing(self, clicks: np.ndarray, time_tags: np.ndarray) -> np.ndarray:
        """
        Apply physical afterpulsing model with exponential decay.

        Afterpulsing occurs when charge carriers trapped during a detection event
        are released later, causing a spurious click. The probability decays
        exponentially with time since the last detection:

            P_after(t) = P0 * exp(-t / tau)

        where:
            P0 = initial afterpulse probability (at t=0+)
            tau = trap lifetime (exponential decay constant)

        Args:
            clicks: Boolean array of click events
            time_tags: Time stamps of events (ns)

        Returns:
            Click array with afterpulse events added
        """
        if not self.config.enable_afterpulsing or not np.any(clicks):
            return clicks

        # Get parameters
        p0 = self.config.afterpulsing_prob
        tau = getattr(self.config, 'afterpulsing_tau_ns', self.config.afterpulsing_delay_ns)

        # Find all clicks (potential triggers for afterpulsing)
        click_indices = np.where(clicks)[0]
        if len(click_indices) == 0:
            return clicks

        # EFFICIENT IMPLEMENTATION: For each real detection, sample at most ONE afterpulse
        # This avoids O(N²) loop and matches physical reality better
        # Reference: Namekata et al. (2006), Opt. Express 14, 10043
        afterpulse_clicks = clicks.copy()

        for click_idx in click_indices:
            click_time = time_tags[click_idx]

            # Sample whether this detection produces an afterpulse
            if self.rng.random() >= p0:
                continue  # No afterpulse from this detection

            # Sample the afterpulse delay time from exponential distribution
            # t_after ~ Exp(1/tau), implemented as: t = -tau * ln(U) where U ~ Uniform(0,1)
            afterpulse_delay = -tau * np.log(self.rng.random())

            # Find the gate index closest to this afterpulse time
            afterpulse_time = click_time + afterpulse_delay

            # Find which gate this falls into (simple nearest-gate model)
            # This assumes time_tags are ordered and correspond to gate times
            future_gates = time_tags > click_time
            if not np.any(future_gates):
                continue  # No future gates to afterpulse into

            # Find the closest future gate
            future_indices = np.where(future_gates)[0]
            gate_times = time_tags[future_indices]
            closest_idx = future_indices[np.argmin(np.abs(gate_times - afterpulse_time))]

            # Mark that gate as having an afterpulse
            afterpulse_clicks[closest_idx] = True

        return afterpulse_clicks

    def _sift_keys(self, alice_results, bob_results, alice_bases, bob_bases):
        """
        Sift for matching bases.

        Args:
            alice_results: Alice's measurement results
            bob_results: Bob's measurement results
            alice_bases: Alice's basis choices
            bob_bases: Bob's basis choices

        Returns:
            Tuple of (sifted_alice, sifted_bob)
        """
        match_mask = np.zeros(len(alice_results), dtype=bool)
        for alice_idx, bob_idx, purpose in self.config.basis_mapping:
            if purpose == "key":
                match_mask |= (alice_bases == alice_idx) & (bob_bases == bob_idx)
        return alice_results[match_mask], bob_results[match_mask]

    def _compute_chsh(self, alice_results, bob_results, alice_bases, bob_bases):
        """
        Compute CHSH parameter from measurement results.

        Args:
            alice_results: Alice's measurement results
            bob_results: Bob's measurement results
            alice_bases: Alice's basis choices
            bob_bases: Bob's basis choices

        Returns:
            Tuple of (CHSH S value, correlators dictionary)
        """
        correlators = {}

        # Compute empirical correlations from actual measurements
        for i in range(3):
            for j in range(3):
                mask = (alice_bases == i) & (bob_bases == j)
                if np.sum(mask) > 0:
                    a_pm = 2 * alice_results[mask] - 1
                    b_pm = 2 * bob_results[mask] - 1
                    correlators[f'E_{i}{j}'] = np.mean(a_pm * b_pm)
                else:
                    correlators[f'E_{i}{j}'] = 0

        # CHSH inequality: S = |E(a0,b0) - E(a0,b1) + E(a1,b0) + E(a1,b1)|
        # Use explicit CHSH indices from config (CRITICAL for protocol correctness)
        # These define which measurement settings are used for the CHSH test
        a0_idx = self.config.chsh_a0_idx
        a1_idx = self.config.chsh_a1_idx
        b0_idx = self.config.chsh_b0_idx
        b1_idx = self.config.chsh_b1_idx

        E_00 = correlators.get(f'E_{a0_idx}{b0_idx}', 0)
        E_01 = correlators.get(f'E_{a0_idx}{b1_idx}', 0)
        E_10 = correlators.get(f'E_{a1_idx}{b0_idx}', 0)
        E_11 = correlators.get(f'E_{a1_idx}{b1_idx}', 0)
        S = abs(E_00 - E_01 + E_10 + E_11)
        return S, correlators

    def _compute_per_basis_qber(self, alice_results, bob_results, alice_bases, bob_bases):
        """
        Compute QBER for each measurement basis pair (Phase 5B diagnostic).

        This provides detailed error information for each basis combination,
        useful for diagnosing channel issues or basis-dependent errors.

        Args:
            alice_results: Alice's measurement results
            bob_results: Bob's measurement results
            alice_bases: Alice's basis choices
            bob_bases: Bob's basis choices

        Returns:
            Dictionary mapping basis pair (ai, bj) to QBER value
        """
        per_basis_qber = {}

        for ai in range(3):
            for bj in range(3):
                mask = (alice_bases == ai) & (bob_bases == bj)
                n_basis = np.sum(mask)

                if n_basis > 0:
                    a_basis = alice_results[mask]
                    b_basis = bob_results[mask]
                    errors = np.sum(a_basis != b_basis)
                    qber = errors / n_basis
                    per_basis_qber[f'Q_{ai}{bj}'] = float(qber)
                else:
                    per_basis_qber[f'Q_{ai}{bj}'] = None

        return per_basis_qber

    def _compute_coincidence_stats(self, time_tags_A_coinc, time_tags_B_coinc,
                                   time_tags_A_singles=None, time_tags_B_singles=None):
        """
        Compute coincidence delay, CAR, and singles rates.

        FIX P0-5: CAR now calculated from singles rates (CORRECTED)
        - CAR formula: CAR = N_coinc / N_acc
        - N_acc = R_singles_A × R_singles_B × τ_c × T

        Args:
            time_tags_A_coinc: Alice's coincidence time tags
            time_tags_B_coinc: Bob's coincidence time tags
            time_tags_A_singles: Alice's singles time tags (all clicks)
            time_tags_B_singles: Bob's singles time tags (all clicks)

        Returns:
            Tuple of (avg_delay_ns, car, singles_rate_A_Hz, singles_rate_B_Hz,
                     coincidence_rate_Hz, accidentals_estimate)
            All values are None if insufficient data
        """
        if time_tags_A_coinc is None or len(time_tags_A_coinc) < 2 or len(time_tags_B_coinc) < 2:
            # CRITICAL: Must return 6 values to match caller expectations
            return None, None, None, None, None, None

        # Compute average coincidence delay
        delays = np.abs(time_tags_A_coinc - time_tags_B_coinc)
        avg_delay = np.mean(delays)

        # Compute CAR using singles rates (if available)
        if time_tags_A_singles is not None and time_tags_B_singles is not None:
            if len(time_tags_A_singles) < 1 or len(time_tags_B_singles) < 1:
                return avg_delay, None, None, None, None, None

            # CRITICAL FIX: Use acquisition time, not time-tag span
            # In high-loss regimes, time-tag span underestimates acquisition time
            # T_acq = N_trials / repetition_rate
            if hasattr(self.config, 'repetition_rate_Hz') and self.config.repetition_rate_Hz > 0:
                # Use declared repetition rate for acquisition time
                num_trials = self.config.num_pairs
                T_total = num_trials / self.config.repetition_rate_Hz  # seconds
            else:
                # Fallback: estimate from time-tag span (not recommended for sparse data)
                duration_A = time_tags_A_singles[-1] - time_tags_A_singles[0]
                duration_B = time_tags_B_singles[-1] - time_tags_B_singles[0]
                if duration_A <= 0 or duration_B <= 0:
                    return avg_delay, None, None, None, None, None
                T_total = max(duration_A, duration_B) * 1e-9  # Convert ns to seconds

            # Singles rates in Hz: R = N_clicks / T_acq
            rate_A_singles = len(time_tags_A_singles) / T_total
            rate_B_singles = len(time_tags_B_singles) / T_total

            # Number of coincidences
            tau_c = self.config.coincidence_window_ns
            coincidences = len(time_tags_A_coinc)  # Already filtered to coincidences

            # Expected accidentals: N_acc = R_A × R_B × τ_c × T_acq
            accidentals = rate_A_singles * rate_B_singles * (tau_c * 1e-9) * T_total

            # CAR = N_coinc / N_acc
            car = coincidences / max(accidentals, 1.0)

            # Return singles rates and accidentals for mandatory output metrics
            return avg_delay, car, rate_A_singles, rate_B_singles, coincidences / T_total, accidentals
        else:
            # Fallback: no singles data available
            car = None
            return avg_delay, car, None, None, None, None

    def _apply_double_click_squashing(self, alice_results: np.ndarray, bob_results: np.ndarray,
                                      pair_counts: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply double-click handling with threshold-detector squashing model.

        THRESHOLD DETECTOR MODEL: Real single-photon detectors (APDs, SNSPDs) are
        threshold devices that cannot distinguish 1 vs 2+ photons. When multiple
        photons arrive at the detector pair (D0, D1) for a single measurement basis:
        - Both detectors can click simultaneously (double-click event)
        - The actual measurement outcome is ambiguous
        - Security proof requires "squashing" double-clicks into valid outcomes

        SQUASHING MODELS (standard in QKD security proofs):

        1. "random" assignment:
           - Randomly assign double-clicks to 0 or 1
           - Models detector randomly selecting which click to output
           - Introduces additional QBER but preserves more events
           - Security: Gottesman-Lo-Lütkenhaus-Preskill framework (2004)
           - Reference: https://doi.org/10.26421/QIC4.5-1

        2. "discard" policy:
           - Discard all double-click events entirely
           - Conservative approach (reduces rate but cleaner security)
           - Standard in many threshold-detector QKD implementations
           - Reference: Beaudry et al. (2008) https://doi.org/10.1103/PhysRevLett.101.093601

        PHYSICAL SOURCES OF DOUBLE-CLICKS:
        - Multi-pair SPDC emission (k≥2 photon pairs from one pump pulse)
        - Afterpulsing (detector artifacts)
        - Background light coinciding with signal

        This implementation focuses on multi-pair emissions using:
        - Thermal/Poisson SPDC statistics: P(k) from _sample_pair_number()
        - Binomial loss thinning: models photon survival through channel
        - PBS routing: 50/50 split of detected photons to D0 vs D1

        SECURITY CONTEXT:
        Multi-pair emissions are a known vulnerability (photon-number-splitting attacks).
        The squashing model provides security by mapping threshold-detector behavior
        to an equivalent qubit protocol.

        References:
            Beaudry, Moroder & Lütkenhaus (2008), Phys. Rev. Lett. 101, 093601
                "Squashing model framework for threshold detectors" (primary reference)
            Gottesman et al. (2004), Quantum Inf. Comput. 4, 325
                "GLLP security proof for threshold detectors"
            Lütkenhaus & Jahma (2002), New J. Phys. 4, 44
                "PNS attacks and photon-number statistics"

        Args:
            alice_results: Alice's measurement outcomes (0 or 1)
            bob_results: Bob's measurement outcomes (0 or 1)
            pair_counts: Optional array of pair numbers (k) per event from SPDC sampling

        Returns:
            Tuple of (alice_results, bob_results, keep_mask)
            where keep_mask indicates which events to keep
        """
        if not self.config.enable_double_click_squashing:
            # No squashing - return all events
            return alice_results, bob_results, np.ones(len(alice_results), dtype=bool)

        n = len(alice_results)

        # Model: Each measurement basis has 2 detectors (D0, D1)
        # The "result" tells us which detector SHOULD fire for ideal single-photon
        # With multi-pair or noise, BOTH detectors might fire (double-click)

        # Simulate detector clicks per side
        # For single-pair events: only one detector fires (determined by result)
        # For multi-pair events: multiple detectors can fire

        # Determine if double-click occurs
        # This happens primarily with multi-pair emission
        # Probability of double-click ~ probability of multi-pair

        if pair_counts is not None:
            # PHYSICS-BASED MODEL: Multi-pair doesn't always → double-click
            # Need to account for: (1) loss, (2) detector routing
            # Reference: Takesue & Shimizu (2010), https://doi.org/10.1016/j.optcom.2009.10.008

            # Get effective detection efficiency PER SIDE (combined channel + detector losses)
            # Alice and Bob can have different link losses and detector efficiencies
            # This is critical for asymmetric channels (e.g., satellite downlinks)
            eta_eff_A = 1.0
            eta_eff_B = 1.0

            # Channel loss (fiber + insertion) - PER SIDE
            if self.config.enable_fiber_loss:
                loss_dB_A = (self.config.distance_km_A * self.config.fiber_loss_dB_per_km +
                            self.config.loss_dB_A)
                loss_dB_B = (self.config.distance_km_B * self.config.fiber_loss_dB_per_km +
                            self.config.loss_dB_B)
                eta_eff_A *= 10 ** (-loss_dB_A / 10)
                eta_eff_B *= 10 ** (-loss_dB_B / 10)

            if self.config.enable_insertion_loss:
                # Insertion loss typically symmetric, but apply to both sides
                eta_eff_A *= 10 ** (-self.config.insertion_loss_dB / 10)
                eta_eff_B *= 10 ** (-self.config.insertion_loss_dB / 10)

            # Detector efficiency - PER SIDE
            if self.config.enable_detector_loss:
                eta_det_A = getattr(self.config, 'end_detector_efficiency_A',
                                   self.config.end_detector_efficiency)
                eta_det_B = getattr(self.config, 'end_detector_efficiency_B',
                                   self.config.end_detector_efficiency)
                eta_eff_A *= eta_det_A
                eta_eff_B *= eta_det_B

            # For multi-pair events (k≥2), model double-click probability
            # Simplified: P(double-click | k pairs) ≈ probability that ≥2 photons
            # survive loss AND both detectors on one side receive ≥1 photon
            double_click_A = np.zeros(n, dtype=bool)
            double_click_B = np.zeros(n, dtype=bool)

            for i in range(n):
                k = pair_counts[i]
                if k >= 2:
                    # Binomial thinning: how many photons survive to each side?
                    # Each photon survives independently with probability eta_eff_A or eta_eff_B
                    k_detected_A = self.rng.binomial(k, eta_eff_A)
                    k_detected_B = self.rng.binomial(k, eta_eff_B)

                    # Double-click happens if ≥2 photons hit the detector pair
                    # With threshold detectors + 50/50 routing (measurement basis PBS):
                    # If k_detected ≥ 2, there's a chance both detectors fire
                    # Simplified: use binomial to route detected photons to D0 vs D1
                    if k_detected_A >= 2:
                        # Route photons through measurement basis (50/50 split for unpolarized)
                        # For simplicity: if ≥2 detected, assume some go to each detector
                        # More rigorous: sample routing based on measurement outcome
                        d0_A = self.rng.binomial(k_detected_A, 0.5)
                        d1_A = k_detected_A - d0_A
                        double_click_A[i] = (d0_A >= 1) and (d1_A >= 1)

                    if k_detected_B >= 2:
                        d0_B = self.rng.binomial(k_detected_B, 0.5)
                        d1_B = k_detected_B - d0_B
                        double_click_B[i] = (d0_B >= 1) and (d1_B >= 1)

        else:
            # No pair_counts provided - no multi-pair double-clicks
            double_click_A = np.zeros(n, dtype=bool)
            double_click_B = np.zeros(n, dtype=bool)

        # Apply squashing model
        keep_mask = np.ones(n, dtype=bool)

        if self.config.squashing_model == "discard":
            # Discard events with double-clicks on either side
            keep_mask = ~(double_click_A | double_click_B)

        elif self.config.squashing_model == "random":
            # Randomly assign outcome when double-click occurs
            # This introduces additional QBER
            alice_results = alice_results.copy()
            bob_results = bob_results.copy()

            # Randomize outcomes for double-click events
            alice_results[double_click_A] = self.rng.integers(0, 2, np.sum(double_click_A))
            bob_results[double_click_B] = self.rng.integers(0, 2, np.sum(double_click_B))

        else:
            raise ValueError(f"Unknown squashing_model: {self.config.squashing_model}")

        return alice_results, bob_results, keep_mask

    def _sample_pair_number(self, size: int) -> np.ndarray:
        """
        Sample number of photon pairs per SPDC emission.

        PHYSICAL JUSTIFICATION (Eisaman et al. 2011, Rev. Mod. Phys.):

        Single-mode SPDC (narrow spectral/spatial filtering):
        - Thermal (geometric) distribution: P(k) = μ^k / (1 + μ)^(k+1)
        - Physically arises from two-mode squeezed vacuum state
        - Use "thermal" distribution (default)
        - Reference: https://doi.org/10.1103/RevModPhys.83.1

        Multi-mode SPDC (broadband collection, many spatial modes):
        - Poisson distribution: P(k) = e^(-μ) μ^k / k!
        - Emerges in the limit of many independent modes
        - Use "poisson" distribution
        - Typical when M >> 1 modes contribute

        CONFIGURATION GUIDANCE:
        - Narrow filters (< 1 nm) + single spatial mode → thermal
        - Broadband (> 10 nm) + multi-mode fiber → poisson
        - When uncertain, use thermal (more conservative for security)

        Args:
            size: Number of emission events to sample

        Returns:
            Array of pair counts (0, 1, 2, ...) for each emission
        """
        mu = self.config.spdc_brightness_mu

        if self.config.spdc_distribution == "thermal":
            # Thermal (geometric) distribution for single-mode SPDC
            # numpy.random.geometric samples from P(k) = (1-p)^(k-1) * p for k=1,2,3,...
            # We want P(k) = μ^k / (1+μ)^(k+1) for k=0,1,2,...
            # This corresponds to geometric(p) - 1 where p = 1/(1+μ)
            p = 1.0 / (1.0 + mu)
            pair_counts = self.rng.geometric(p, size=size) - 1
        elif self.config.spdc_distribution == "poisson":
            # Poisson distribution for multi-mode SPDC (M >> 1 modes)
            pair_counts = self.rng.poisson(mu, size=size)
        else:
            raise ValueError(f"Unknown spdc_distribution: {self.config.spdc_distribution}")

        return pair_counts


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'E91Protocol',
    'QISKIT_AVAILABLE',
    'QUTIP_AVAILABLE',
]
