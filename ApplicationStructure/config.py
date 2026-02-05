"""
================================================================================
E91 QKD Configuration File
================================================================================

This file contains all configurable parameters for the E91 Quantum Key
Distribution simulator. Parameters are organized by category for easy
modification and understanding.

Author: Tyler Barr
Version: 7.0.0 Refactored
Date: 2025

================================================================================
REFERENCES:
- E91 Original Paper: Ekert, A. K. (1991). Physical Review Letters, 67(6), 661
- CHSH Inequality: Clauser et al. (1969). Physical Review Letters, 23(15), 880
- QKD Security: Shor & Preskill (2000). Physical Review Letters, 85(2), 441
================================================================================
"""

import numpy as np
from typing import List, Tuple, Optional

# ============================================================================
# VERSION AND METADATA
# ============================================================================

VERSION = "7.0.0 Professional Refactored"
AUTHOR = "QKD Research Team"
LICENSE = "MIT"

# ============================================================================
# FUNDAMENTAL QUANTUM CONSTANTS
# ============================================================================

class QuantumConstants:
    """Fundamental constants used in quantum key distribution."""

    # CHSH Inequality Bounds
    # Reference: Clauser et al. (1969), PRL 23(15), 880
    # Classical local realistic theories: S ≤ 2
    # Quantum mechanics maximum (Tsirelson bound): S ≤ 2√2 ≈ 2.828
    CHSH_CLASSICAL_BOUND = 2.0
    CHSH_QUANTUM_MAX = 2.0 * np.sqrt(2)  # Tsirelson bound ≈ 2.828

    # QBER (Quantum Bit Error Rate) Threshold
    # Reference: Shor & Preskill (2000), PRL 85(2), 441
    # Security threshold for E91 protocol (approximately 11%)
    QBER_THRESHOLD = 0.11

    # Speed of light (m/s)
    SPEED_OF_LIGHT = 299792458.0

    # Planck's constant (J·s)
    PLANCK_CONSTANT = 6.62607015e-34


# ============================================================================
# DEFAULT MEASUREMENT ANGLES
# ============================================================================

class MeasurementAngles:
    """
    Default measurement angles for Alice and Bob.

    MATHEMATICAL FOUNDATION:
    ------------------------
    For optimal CHSH violation, angles are chosen to maximize:
    S = |E(a₀,b₀) - E(a₀,b₁) + E(a₁,b₀) + E(a₁,b₁)|

    Where E(aᵢ,bⱼ) = P(++|aᵢ,bⱼ) + P(--|aᵢ,bⱼ) - P(+-|aᵢ,bⱼ) - P(-+|aᵢ,bⱼ)

    Optimal configuration (Tsirelson):
    - Alice: {0°, 22.5°, 45°} = {0, π/8, π/4}
    - Bob:   {22.5°, 45°, 67.5°} = {π/8, π/4, 3π/8}

    This gives S = 2√2 ≈ 2.828 (maximum quantum violation)

    Reference: Tsirelson (1980), Letters in Mathematical Physics, 4(2), 93-100
    """

    # Alice's measurement angles (radians)
    # a₀ = 0°, a₁ = 22.5°, a₂ = 45°
    ALICE_ANGLES_DEFAULT = [0.0, np.pi/8, np.pi/4]

    # Bob's measurement angles (radians)
    # b₀ = 22.5°, b₁ = 45°, b₂ = 67.5°
    BOB_ANGLES_DEFAULT = [np.pi/8, np.pi/4, 3*np.pi/8]

    # Alternative configurations for testing
    ALICE_ANGLES_SYMMETRIC = [0.0, np.pi/4, np.pi/2]
    BOB_ANGLES_SYMMETRIC = [np.pi/4, np.pi/2, 3*np.pi/4]


# ============================================================================
# BELL STATES
# ============================================================================

class BellStates:
    """
    The four maximally entangled Bell states.

    MATHEMATICAL DEFINITION:
    ------------------------
    |Φ⁺⟩ = (|00⟩ + |11⟩)/√2    (phi_plus)
    |Φ⁻⟩ = (|00⟩ - |11⟩)/√2    (phi_minus)
    |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2    (psi_plus)
    |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2    (psi_minus)

    Properties:
    - All are maximally entangled (entanglement entropy = 1)
    - Form a complete orthonormal basis for two-qubit states
    - Related by local unitary transformations

    Reference: Bennett et al. (1993), PRL 70(13), 1895
    """

    PHI_PLUS = "phi_plus"      # Most commonly used in E91
    PHI_MINUS = "phi_minus"
    PSI_PLUS = "psi_plus"
    PSI_MINUS = "psi_minus"

    ALL_STATES = [PHI_PLUS, PHI_MINUS, PSI_PLUS, PSI_MINUS]
    DEFAULT = PHI_PLUS


# ============================================================================
# SIMULATION BACKEND OPTIONS
# ============================================================================

class BackendConfig:
    """Configuration for quantum simulation backends."""

    # Available backends (Qiskit or QuTiP REQUIRED)
    QISKIT = "qiskit"      # IBM Qiskit (required - full statevector simulation)
    QUTIP = "qutip"        # QuTiP (required - density matrix simulation)

    # Default backend
    DEFAULT = QISKIT

    # Backend-specific settings
    QISKIT_METHOD = "statevector"  # Simulation method
    QUTIP_SOLVER = "master"        # Master equation solver


# ============================================================================
# CORE EXPERIMENT PARAMETERS
# ============================================================================

class CoreParameters:
    """Core parameters for running E91 experiments."""

    # Number of entangled pairs to generate
    # Typical range: 1,000 - 1,000,000
    # More pairs = better statistics but slower simulation
    NUM_PAIRS_DEFAULT = 10000
    NUM_PAIRS_MIN = 100
    NUM_PAIRS_MAX = 10000000

    # Random seed for reproducibility
    # Set to specific value for deterministic results
    # Set to None for true randomness
    SEED_DEFAULT = 42

    # Basis selection probabilities (uniform if None)
    # Must sum to 1.0 if specified
    # Example: [0.5, 0.25, 0.25] favors first angle
    ALICE_BASIS_PROBS_DEFAULT = None
    BOB_BASIS_PROBS_DEFAULT = None


# ============================================================================
# SECURITY PARAMETERS
# ============================================================================

class SecurityParameters:
    """
    Security parameters for finite-size key extraction.

    MATHEMATICAL FOUNDATION:
    ------------------------
    The finite-size secret key length is calculated as:

    ℓ = n - leak_EC - leak_PA

    Where:
    - n: number of sifted bits
    - leak_EC: information leaked during error correction
    - leak_PA: privacy amplification overhead

    leak_EC = f_EC · n · H(Q_μ + δ)
    leak_PA = log₂(1/ε_sec) + log₂(1/ε_cor)

    Where:
    - f_EC: error correction efficiency factor
    - H(x): binary entropy function = -x·log₂(x) - (1-x)·log₂(1-x)
    - Q_μ: measured QBER
    - δ: statistical fluctuation = √[ln(2/ε_sec)/(2n_PE)]
    - ε_sec: secrecy failure probability
    - ε_cor: correctness failure probability
    - n_PE: number of bits used for parameter estimation

    Reference: Tomamichel et al. (2012), Nature Communications 3, 634
    """

    # Security failure probability (ε_sec)
    # Probability that the key is not ε-secure
    # Typical value: 10⁻⁹ (one in a billion)
    EPSILON_SEC_DEFAULT = 1e-9
    EPSILON_SEC_MIN = 1e-15
    EPSILON_SEC_MAX = 1e-6

    # Correctness failure probability (ε_cor)
    # Probability of error correction failure
    # Typical value: 10⁻¹⁵ (extremely low)
    EPSILON_COR_DEFAULT = 1e-15
    EPSILON_COR_MIN = 1e-20
    EPSILON_COR_MAX = 1e-10

    # Parameter estimation fraction
    # Fraction of sifted bits used for QBER estimation
    # Remaining bits used for key generation
    # Typical range: 0.1 - 0.5 (10% - 50%)
    PE_FRACTION_DEFAULT = 0.1
    PE_FRACTION_MIN = 0.01
    PE_FRACTION_MAX = 0.9

    # Error correction efficiency (f_EC)
    # Real error correction codes have efficiency > 1
    # Shannon limit: f_EC = 1.0
    # Practical codes: f_EC ≈ 1.1 - 1.2
    F_EC_DEFAULT = 1.2
    F_EC_MIN = 1.0
    F_EC_MAX = 2.0

    # Information reconciliation protocol
    IR_BBBSS = "BBBSS"      # Bennett-Brassard-Robert (1988)
    IR_CASCADE = "CASCADE"  # Cascade protocol (Brassard & Salvail, 1993)
    IR_WINNOW = "WINNOW"    # Winnow protocol (Buttler et al., 2003)
    IR_LDPC = "LDPC"        # Low-density parity check codes
    IR_DEFAULT = IR_BBBSS

    # Block length for error correction
    # Typical range: 256 - 4096 bits
    BLOCK_LENGTH_DEFAULT = 1024
    BLOCK_LENGTH_MIN = 64
    BLOCK_LENGTH_MAX = 65536

    # Code rate (ratio of data bits to total bits)
    # Lower rate = more error correction capability
    # Typical range: 0.3 - 0.9
    CODE_RATE_DEFAULT = 0.5
    CODE_RATE_MIN = 0.1
    CODE_RATE_MAX = 0.9


# ============================================================================
# DEVICE-INDEPENDENT (DI) PARAMETERS
# ============================================================================

class DeviceIndependentParameters:
    """
    Parameters for Device-Independent Quantum Key Distribution.

    MATHEMATICAL FOUNDATION:
    ------------------------
    Device-independent QKD bases security on violation of Bell inequalities,
    without assuming anything about the internal workings of devices.

    The key rate bound for DI-QKD is:

    r_DI ≥ H(A|E) - H(A|B)

    Where:
    - H(A|E): conditional entropy of Alice's outcome given Eve's information
    - H(A|B): conditional entropy between Alice and Bob (related to QBER)

    Lower bound on H(A|E) from CHSH violation S:
    H(A|E) ≥ h((1 + √((S/2)² - 1))/2)

    Where h(x) is the binary entropy function.

    For routed DI-QKD with imperfect detectors:
    S_eff = S / √(2η_eff)

    Where η_eff is the effective detection efficiency.

    Reference:
    - Acín et al. (2007), PRL 98, 230501
    - Vazirani & Vidick (2014), PRL 113, 140501
    """

    # DI bound type
    DI_BOUND_DEFAULT = "default_di_bound"
    DI_BOUND_ENTROPY = "entropy_bound"
    DI_BOUND_NUMERICAL = "numerical_bound"

    # Enable routed DI-QKD
    # Accounts for photon routing losses
    ENABLE_ROUTED_DI_DEFAULT = False

    # Routing efficiency factors
    # η_f: fiber routing efficiency (typically 0.90 - 0.99)
    # η_d: detector routing efficiency (typically 0.85 - 0.95)
    ROUTING_EFFICIENCY_F_DEFAULT = 0.95
    ROUTING_EFFICIENCY_F_MIN = 0.5
    ROUTING_EFFICIENCY_F_MAX = 1.0

    ROUTING_EFFICIENCY_D_DEFAULT = 0.90
    ROUTING_EFFICIENCY_D_MIN = 0.5
    ROUTING_EFFICIENCY_D_MAX = 1.0


# ============================================================================
# NOISE MODEL PARAMETERS
# ============================================================================

class NoiseParameters:
    """
    Parameters for various noise models.

    MATHEMATICAL FOUNDATION:
    ------------------------
    1. DEPOLARIZING NOISE:
       ρ_out = (1-α)ρ_in + (α/3)(XρX + YρY + ZρZ)

       Where α is the depolarizing probability, and X,Y,Z are Pauli matrices.
       Effect: Reduces visibility and increases QBER

    2. VISIBILITY REDUCTION:
       ρ_mixed = V·ρ_pure + (1-V)·(I/4)

       Where V is the visibility (0 ≤ V ≤ 1), I is identity matrix
       Perfect visibility V=1: pure entangled state
       No visibility V=0: completely mixed state

    3. INTRINSIC ERROR:
       Bit-flip probability e_d per qubit
       Effective QBER ≈ e_d (for small e_d)

    Reference: Nielsen & Chuang (2010), "Quantum Computation and Quantum
               Information", Chapter 8
    """

    # === Depolarizing Noise ===
    # Simulates white noise from environmental decoherence
    ENABLE_DEPOLARIZING_DEFAULT = False
    DEPOLARIZING_ALPHA_DEFAULT = 0.06  # 6% depolarizing probability
    DEPOLARIZING_ALPHA_MIN = 0.0
    DEPOLARIZING_ALPHA_MAX = 1.0

    # === Visibility Reduction ===
    # Simulates imperfect state preparation or partial decoherence
    ENABLE_VISIBILITY_DEFAULT = False
    VISIBILITY_DEFAULT = 0.98  # 98% visibility
    VISIBILITY_MIN = 0.0
    VISIBILITY_MAX = 1.0

    # === Intrinsic Error ===
    # Simulates detector and readout errors
    ENABLE_INTRINSIC_ERROR_DEFAULT = False
    INTRINSIC_ERROR_ED_DEFAULT = 0.01  # 1% error rate
    INTRINSIC_ERROR_ED_MIN = 0.0
    INTRINSIC_ERROR_ED_MAX = 0.5


# ============================================================================
# CHANNEL LOSS PARAMETERS
# ============================================================================

class ChannelLossParameters:
    """
    Parameters for optical channel losses.

    MATHEMATICAL FOUNDATION:
    ------------------------
    1. FIBER ATTENUATION:
       P_out = P_in · 10^(-α·L/10)

       Where:
       - P_out: output power
       - P_in: input power
       - α: attenuation coefficient (dB/km)
       - L: fiber length (km)

       For standard telecom fiber at 1550nm: α ≈ 0.2 dB/km
       At 810nm (typical SPDC): α ≈ 2-3 dB/km

    2. INSERTION LOSS:
       L_insert = 10·log₁₀(P_out/P_in) (dB)

       Typical values:
       - Beam splitter: 3 dB (50/50 split)
       - Optical switch: 0.5-2 dB
       - Connector: 0.1-0.5 dB

    3. POLARIZATION MISALIGNMENT:
       Transmission = cos²(θ)

       Where θ is the angular misalignment
       Loss (dB) = -10·log₁₀(cos²(θ))

    Reference: Agrawal (2012), "Fiber-Optic Communication Systems", 4th ed.
    """

    # === Fiber Loss ===
    ENABLE_FIBER_LOSS_DEFAULT = False

    # Distance from source to Alice (km)
    DISTANCE_KM_A_DEFAULT = 5.0
    DISTANCE_KM_A_MIN = 0.0
    DISTANCE_KM_A_MAX = 1000.0

    # Distance from source to Bob (km)
    DISTANCE_KM_B_DEFAULT = 5.0
    DISTANCE_KM_B_MIN = 0.0
    DISTANCE_KM_B_MAX = 1000.0

    # Fiber attenuation coefficient (dB/km)
    FIBER_LOSS_DB_PER_KM_DEFAULT = 0.2  # Standard SMF-28 at 1550nm
    FIBER_LOSS_DB_PER_KM_MIN = 0.0
    FIBER_LOSS_DB_PER_KM_MAX = 10.0

    # Additional loss at Alice's end (dB)
    LOSS_DB_A_DEFAULT = 0.0
    LOSS_DB_A_MIN = 0.0
    LOSS_DB_A_MAX = 50.0

    # Additional loss at Bob's end (dB)
    LOSS_DB_B_DEFAULT = 0.0
    LOSS_DB_B_MIN = 0.0
    LOSS_DB_B_MAX = 50.0

    # === Insertion Loss ===
    ENABLE_INSERTION_LOSS_DEFAULT = False
    INSERTION_LOSS_DB_DEFAULT = 0.5
    INSERTION_LOSS_DB_MIN = 0.0
    INSERTION_LOSS_DB_MAX = 10.0

    # === Polarization Misalignment ===
    ENABLE_MISALIGNMENT_DEFAULT = False
    POLARIZATION_DRIFT_DEG_DEFAULT = 2.0  # 2 degrees drift
    POLARIZATION_DRIFT_DEG_MIN = 0.0
    POLARIZATION_DRIFT_DEG_MAX = 90.0

    # === Quantum Repeaters ===
    # Model for compensating long-distance losses
    ENABLE_REPEATERS_DEFAULT = False
    NUM_REPEATERS_DEFAULT = 0
    NUM_REPEATERS_MIN = 0
    NUM_REPEATERS_MAX = 100
    REPEATER_GAIN_DB_DEFAULT = 0.0
    REPEATER_GAIN_DB_MIN = 0.0
    REPEATER_GAIN_DB_MAX = 50.0


# ============================================================================
# DETECTOR PARAMETERS
# ============================================================================

class DetectorParameters:
    """
    Parameters for single-photon detector models.

    MATHEMATICAL FOUNDATION:
    ------------------------
    1. DETECTION EFFICIENCY:
       η_total = η_herald · η_detector

       Where:
       - η_herald: heralding efficiency (photon makes it to detector)
       - η_detector: quantum efficiency (photon detected when present)

       Typical values for SNSPDs: 80-95%
       Typical values for APDs: 50-70%

    2. DARK COUNT RATE:
       P_dark = 1 - exp(-λ_dark · Δt)

       Where:
       - λ_dark: dark count rate (counts per second)
       - Δt: detection time window (seconds)

       Typical values:
       - SNSPDs: 10-1000 cps
       - APDs: 100-10000 cps

    3. TIMING JITTER:
       Affects coincidence window size
       Gaussian distribution with σ = jitter_time
       Total timing uncertainty = √(σ_A² + σ_B²)

       Typical values:
       - SNSPDs: 50-150 ps
       - APDs: 300-500 ps

    4. AFTERPULSING:
       P_after(t) = P₀ · exp(-t/τ_trap)

       Where:
       - P₀: initial afterpulsing probability
       - τ_trap: trap lifetime
       - t: time after detection

    Reference: Hadfield (2009), Nature Photonics 3, 696-705
    """

    # === Detection Efficiency ===
    ENABLE_DETECTOR_LOSS_DEFAULT = False

    # Heralding efficiency (photon routing to detector)
    HERALDING_EFFICIENCY_DEFAULT = 0.9
    HERALDING_EFFICIENCY_MIN = 0.0
    HERALDING_EFFICIENCY_MAX = 1.0

    # Detector quantum efficiency
    END_DETECTOR_EFFICIENCY_DEFAULT = 0.8
    END_DETECTOR_EFFICIENCY_MIN = 0.0
    END_DETECTOR_EFFICIENCY_MAX = 1.0

    # === Dark Counts ===
    ENABLE_DARK_COUNTS_DEFAULT = False

    # Dark count rate (counts per second)
    DARK_CPS_DEFAULT = 100.0
    DARK_CPS_MIN = 0.0
    DARK_CPS_MAX = 100000.0

    # Dark count probability per detection window
    DARK_PROB_DEFAULT = 0.001
    DARK_PROB_MIN = 0.0
    DARK_PROB_MAX = 0.1

    # Use dark counts per second (True) or probability (False)
    USE_DARK_CPS_DEFAULT = False

    # === Timing Jitter ===
    ENABLE_TIMING_JITTER_DEFAULT = False
    JITTER_NS_DEFAULT = 0.5  # 500 picoseconds
    JITTER_NS_MIN = 0.0
    JITTER_NS_MAX = 10.0

    # === Dead Time ===
    # Time detector is inactive after a detection
    ENABLE_DEADTIME_DEFAULT = False
    DEADTIME_NS_DEFAULT = 50.0  # 50 nanoseconds
    DEADTIME_NS_MIN = 0.0
    DEADTIME_NS_MAX = 10000.0

    # === Afterpulsing ===
    ENABLE_AFTERPULSING_DEFAULT = False
    AFTERPULSING_PROB_DEFAULT = 0.05  # 5% probability
    AFTERPULSING_PROB_MIN = 0.0
    AFTERPULSING_PROB_MAX = 0.5
    AFTERPULSING_DELAY_NS_DEFAULT = 100.0
    AFTERPULSING_DELAY_NS_MIN = 10.0
    AFTERPULSING_DELAY_NS_MAX = 10000.0

    # === Saturation ===
    # Maximum count rate before detector saturates
    ENABLE_SATURATION_DEFAULT = False
    SATURATION_RATE_DEFAULT = 1e6  # 1 MHz
    SATURATION_RATE_MIN = 1e3
    SATURATION_RATE_MAX = 1e9


# ============================================================================
# BACKGROUND AND NOISE PHOTONS
# ============================================================================

class BackgroundParameters:
    """
    Parameters for background photon noise.

    MATHEMATICAL FOUNDATION:
    ------------------------
    Background photons create accidental coincidences that increase QBER:

    N_acc = 2 · R_A · R_B · Δt

    Where:
    - R_A, R_B: background count rates at Alice and Bob
    - Δt: coincidence time window

    The background yield Y₀:
    Y₀ = N_acc / N_total

    Effective QBER contribution:
    Q_background ≈ Y₀ / 2 (assumes random background)

    Reference: Scarani et al. (2009), RMP 81, 1301
    """

    ENABLE_BACKGROUND_DEFAULT = False

    # Background yield (fraction of coincidences from background)
    Y0_DEFAULT = 1e-6
    Y0_MIN = 0.0
    Y0_MAX = 0.1

    # Background count rate (counts per second)
    BACKGROUND_CPS_DEFAULT = 500.0
    BACKGROUND_CPS_MIN = 0.0
    BACKGROUND_CPS_MAX = 1000000.0

    # Coincidence time window (nanoseconds)
    COINCIDENCE_WINDOW_NS_DEFAULT = 1.0
    COINCIDENCE_WINDOW_NS_MIN = 0.01
    COINCIDENCE_WINDOW_NS_MAX = 1000.0

    # === Time Tagging ===
    ENABLE_TIME_TAGGING_DEFAULT = False

    # Pulse repetition rate (Hz)
    REPETITION_RATE_HZ_DEFAULT = 1e6  # 1 MHz
    REPETITION_RATE_HZ_MIN = 1e3
    REPETITION_RATE_HZ_MAX = 1e9


# ============================================================================
# PHOTON SOURCE PARAMETERS
# ============================================================================

class SourceParameters:
    """
    Parameters for entangled photon pair sources.

    MATHEMATICAL FOUNDATION:
    ------------------------
    1. SPDC (Spontaneous Parametric Down-Conversion):
       Mean photon pair number per pulse:
       μ = η_SPDC · P_pump · L_crystal / (hν · A_eff)

       Where:
       - η_SPDC: SPDC efficiency
       - P_pump: pump power
       - L_crystal: crystal length
       - hν: photon energy
       - A_eff: effective mode area

    2. MULTI-PAIR EMISSION (Poissonian):
       P(n pairs) = (μⁿ/n!) · exp(-μ)

       For small μ:
       - P(0) ≈ 1-μ (no pairs - loss event)
       - P(1) ≈ μ (desired: single pair)
       - P(2) ≈ μ²/2 (multi-pair - causes errors)

       Optimal operating point: μ ≈ 0.1 - 0.3

    3. SPECTRAL FILTERING:
       Filter bandwidth determines timing resolution and purity
       Narrower filter: better spectral purity, worse timing
       Wider filter: better timing, worse spectral purity

    Reference:
    - Kwiat et al. (1995), PRL 75, 4337
    - Tanzilli et al. (2005), Euro. Phys. J. D 36, 203
    """

    # === Multi-Pair Emission ===
    ENABLE_MULTI_PAIR_DEFAULT = False

    # Mean photon pair number per pump pulse
    SPDC_BRIGHTNESS_MU_DEFAULT = 0.1
    SPDC_BRIGHTNESS_MU_MIN = 0.001
    SPDC_BRIGHTNESS_MU_MAX = 2.0

    # Photon pair generation rate (pairs/second)
    PAIR_RATE_DEFAULT = 1e6  # 1 MHz
    PAIR_RATE_MIN = 1e3
    PAIR_RATE_MAX = 1e9

    # === Pump Configuration ===
    PUMP_POWER_MW_DEFAULT = 10.0  # 10 milliwatts
    PUMP_POWER_MW_MIN = 0.1
    PUMP_POWER_MW_MAX = 1000.0

    # === Wavelength and Filtering ===
    WAVELENGTH_NM_DEFAULT = 810.0  # Standard SPDC at 810nm
    WAVELENGTH_NM_MIN = 400.0
    WAVELENGTH_NM_MAX = 1600.0

    # Spectral filter bandwidth (nm)
    FILTER_BANDWIDTH_NM_DEFAULT = 3.0
    FILTER_BANDWIDTH_NM_MIN = 0.1
    FILTER_BANDWIDTH_NM_MAX = 100.0


# ============================================================================
# SATELLITE QKD PARAMETERS
# ============================================================================

class SatelliteParameters:
    """
    Parameters for satellite-based QKD.

    MATHEMATICAL FOUNDATION:
    ------------------------
    1. FREE-SPACE LINK BUDGET:
       P_rx = P_tx · G_tx · G_rx · (λ/(4πR))² · η_atm

       Where:
       - P_rx: received power
       - P_tx: transmitted power
       - G_tx: transmitter antenna gain
       - G_rx: receiver antenna gain
       - λ: wavelength
       - R: slant range
       - η_atm: atmospheric transmission

    2. BEAM DIVERGENCE:
       θ_div ≈ λ/D_tx (diffraction limit)

       Spot size at receiver:
       d_spot = R · θ_div

       For R=500km, λ=800nm, D=0.3m:
       θ_div ≈ 2.7 μrad → d_spot ≈ 1.35 m

    3. ATMOSPHERIC TURBULENCE:
       Fried parameter r₀ characterizes turbulence strength

       Scintillation index:
       σ_I² = 0.3 · C_n² · k^(7/6) · R^(11/6)

       Where:
       - C_n²: refractive index structure parameter
       - k: wave number = 2π/λ
       - R: propagation distance

    4. POINTING LOSS:
       For Gaussian beams with pointing error δθ:
       Loss = exp(-2(δθ/θ_div)²)

    Reference:
    - Liorni et al. (2017), New J. Phys. 19, 023050
    - Vasylyev et al. (2016), PRL 117, 090501
    """

    ENABLE_SATELLITE_DEFAULT = False

    # === Beam Parameters ===
    # Beam divergence (microradians)
    BEAM_DIVERGENCE_URAD_DEFAULT = 10.0
    BEAM_DIVERGENCE_URAD_MIN = 1.0
    BEAM_DIVERGENCE_URAD_MAX = 100.0

    # Pointing jitter RMS (microradians)
    POINTING_JITTER_URAD_DEFAULT = 5.0
    POINTING_JITTER_URAD_MIN = 0.1
    POINTING_JITTER_URAD_MAX = 50.0

    # Receiver field of view (microradians)
    RECEIVER_FOV_URAD_DEFAULT = 100.0
    RECEIVER_FOV_URAD_MIN = 10.0
    RECEIVER_FOV_URAD_MAX = 1000.0

    # === Link Geometry ===
    # Slant range (km) - distance from satellite to ground
    SLANT_RANGE_KM_DEFAULT = 500.0
    SLANT_RANGE_KM_MIN = 200.0
    SLANT_RANGE_KM_MAX = 2000.0

    # === Optical System ===
    # Transmitter aperture diameter (meters)
    TRANSMITTER_APERTURE_M_DEFAULT = 0.3
    TRANSMITTER_APERTURE_M_MIN = 0.05
    TRANSMITTER_APERTURE_M_MAX = 2.0

    # Receiver aperture diameter (meters)
    RECEIVER_APERTURE_M_DEFAULT = 1.0
    RECEIVER_APERTURE_M_MIN = 0.1
    RECEIVER_APERTURE_M_MAX = 10.0

    # === Background Light ===
    IS_DAYTIME_DEFAULT = False
    SATELLITE_BACKGROUND_CPS_DEFAULT = 1000.0
    SATELLITE_BACKGROUND_CPS_MIN = 10.0
    SATELLITE_BACKGROUND_CPS_MAX = 1000000.0

    # === Atmospheric Turbulence ===
    ENABLE_TURBULENCE_DEFAULT = False

    # Refractive index structure parameter (m^(-2/3))
    CN2_DEFAULT = 1e-15  # Good seeing conditions
    CN2_MIN = 1e-17      # Excellent seeing
    CN2_MAX = 1e-13      # Poor seeing


# ============================================================================
# EAVESDROPPER (EVE) PARAMETERS
# ============================================================================

class EavesdropperParameters:
    """
    Parameters for eavesdropping attacks.

    MATHEMATICAL FOUNDATION:
    ------------------------
    1. INTERCEPT-RESEND ATTACK:
       Eve intercepts photon, measures it, prepares new state

       Induced QBER:
       Q_Eve = (sin²(Δθ/2)) · P_intercept

       Where Δθ is the angle mismatch between bases

       Detection: Increases QBER and reduces CHSH violation

    2. BEAM-SPLITTER ATTACK:
       Eve taps fraction t of the light

       Information gain:
       I(A:E) ≈ -log₂(1-t)

       Disturbance: Lower than intercept-resend

    3. PHOTON-NUMBER-SPLITTING (PNS):
       Eve exploits multi-photon pulses
       Only effective when μ > 0.1

    Reference:
    - Fuchs et al. (1997), PRA 56, 1163
    - Lütkenhaus (2000), PRA 61, 052304
    """

    ENABLE_EAVESDROPPER_DEFAULT = False

    # Attack models
    EVE_MODEL_INTERCEPT_RESEND = "intercept_resend"
    EVE_MODEL_BEAM_SPLITTER = "beam_splitter"
    EVE_MODEL_PNS = "photon_number_splitting"
    EVE_MODEL_DEFAULT = EVE_MODEL_INTERCEPT_RESEND

    # Interception probability (0.0 - 1.0)
    EVE_INTERCEPT_PROB_DEFAULT = 0.0
    EVE_INTERCEPT_PROB_MIN = 0.0
    EVE_INTERCEPT_PROB_MAX = 1.0


# ============================================================================
# NETWORK AND COMMUNICATION PARAMETERS
# ============================================================================

class NetworkParameters:
    """
    Parameters for network-based two-way communication.

    Used for live quantum-secured messaging between computers.
    """

    # Port range for UDP communication
    # Ports 1024-49151 are registered ports (safe for user applications)
    # Ports 49152-65535 are dynamic/private ports
    PORT_MIN = 1024
    PORT_MAX = 65535

    # Default ports for Alice and Bob
    DEFAULT_LOCAL_PORT = 8765
    DEFAULT_PEER_PORT = 8766

    # Default IP addresses
    DEFAULT_LOCALHOST = "127.0.0.1"
    DEFAULT_PEER_HOST = "127.0.0.1"

    # Auto-refresh interval for listener (seconds)
    AUTO_REFRESH_INTERVAL_SEC = 2.0
    AUTO_REFRESH_INTERVAL_MIN = 1.0
    AUTO_REFRESH_INTERVAL_MAX = 10.0

    # UDP socket timeout (seconds)
    SOCKET_TIMEOUT_SEC = 2.0


# ============================================================================
# OUTPUT AND VISUALIZATION PARAMETERS
# ============================================================================

class OutputParameters:
    """Parameters for simulation output and visualization."""

    # Output directory
    OUTPUT_DIRECTORY_DEFAULT = "./results"

    # Enable plot generation
    ENABLE_PLOTS_DEFAULT = True

    # Plot format
    PLOT_FORMAT_PNG = "png"
    PLOT_FORMAT_PDF = "pdf"
    PLOT_FORMAT_SVG = "svg"
    PLOT_FORMAT_DEFAULT = PLOT_FORMAT_PNG

    # Enable structured data export (JSON)
    ENABLE_STRUCTURED_EXPORT_DEFAULT = False

    # Enable BB84 comparison
    ENABLE_BB84_COMPARISON_DEFAULT = False

    # Enable quantum state tomography
    ENABLE_TOMOGRAPHY_DEFAULT = False


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

class PresetConfigurations:
    """
    Predefined experiment configurations for common scenarios.

    These presets provide starting points for typical E91 QKD experiments.
    """

    IDEAL = "Ideal (No Noise)"
    REALISTIC_LAB = "Realistic Lab"
    LONG_DISTANCE = "Long Distance (100km)"
    SATELLITE = "Satellite Link"
    NOISY_CHANNEL = "Noisy Channel"
    CUSTOM = "Custom"

    ALL_PRESETS = [
        IDEAL,
        REALISTIC_LAB,
        LONG_DISTANCE,
        SATELLITE,
        NOISY_CHANNEL,
        CUSTOM
    ]

    DEFAULT = CUSTOM


# ============================================================================
# PHYSICAL WAVELENGTH BANDS
# ============================================================================

class WavelengthBands:
    """Standard wavelength bands for QKD."""

    # Visible spectrum
    VISIBLE_RED = 633.0      # HeNe laser wavelength
    VISIBLE_NIR = 810.0      # Common SPDC wavelength

    # Telecom bands
    O_BAND = 1310.0          # Original band
    C_BAND = 1550.0          # Conventional band (lowest loss)
    L_BAND = 1625.0          # Long wavelength band

    # Default for different applications
    LAB_DEFAULT = VISIBLE_NIR      # Lab experiments
    TELECOM_DEFAULT = C_BAND       # Long-distance fiber
    SATELLITE_DEFAULT = VISIBLE_NIR # Free-space links


# ============================================================================
# VALIDATION RANGES
# ============================================================================

class ValidationRanges:
    """Valid ranges for parameter validation."""

    # Probabilities must be in [0, 1]
    PROBABILITY_MIN = 0.0
    PROBABILITY_MAX = 1.0

    # Angles in radians [0, 2π]
    ANGLE_MIN = 0.0
    ANGLE_MAX = 2 * np.pi

    # Positive values only
    POSITIVE_MIN = 0.0
    POSITIVE_MAX = float('inf')

    # Efficiencies in [0, 1]
    EFFICIENCY_MIN = 0.0
    EFFICIENCY_MAX = 1.0


# ============================================================================
# ERROR MESSAGES
# ============================================================================

class ErrorMessages:
    """Standard error messages for validation."""

    INVALID_PROBABILITY = "Value must be between 0 and 1"
    INVALID_ANGLE = "Angle must be between 0 and 2π radians"
    INVALID_POSITIVE = "Value must be positive"
    INVALID_EFFICIENCY = "Efficiency must be between 0 and 1"
    INVALID_BACKEND = "Invalid backend selection"
    INVALID_BELL_STATE = "Invalid Bell state"
    INVALID_IR_PROTOCOL = "Invalid information reconciliation protocol"
    PROBABILITIES_NOT_NORMALIZED = "Probabilities must sum to 1.0"
    INSUFFICIENT_PAIRS = "Number of pairs too small for statistical significance"
    QBER_TOO_HIGH = "QBER exceeds security threshold"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def db_to_linear(db: float) -> float:
    """
    Convert decibels to linear scale.

    Formula: Linear = 10^(dB/10)

    Args:
        db: Value in decibels

    Returns:
        Value in linear scale
    """
    return 10.0 ** (db / 10.0)


def linear_to_db(linear: float) -> float:
    """
    Convert linear scale to decibels.

    Formula: dB = 10·log₁₀(Linear)

    Args:
        linear: Value in linear scale

    Returns:
        Value in decibels
    """
    if linear <= 0:
        return float('-inf')
    return 10.0 * np.log10(linear)


def transmission_to_db_loss(transmission: float) -> float:
    """
    Convert transmission (0-1) to loss in dB.

    Formula: Loss(dB) = -10·log₁₀(Transmission)

    Args:
        transmission: Transmission coefficient (0-1)

    Returns:
        Loss in decibels (positive value)
    """
    if transmission <= 0:
        return float('inf')
    if transmission >= 1:
        return 0.0
    return -10.0 * np.log10(transmission)


# ============================================================================
# EXPORT CONFIGURATION
# ============================================================================

__all__ = [
    # Constants
    'QuantumConstants',
    'MeasurementAngles',
    'BellStates',
    'BackendConfig',

    # Parameter classes
    'CoreParameters',
    'SecurityParameters',
    'DeviceIndependentParameters',
    'NoiseParameters',
    'ChannelLossParameters',
    'DetectorParameters',
    'BackgroundParameters',
    'SourceParameters',
    'SatelliteParameters',
    'EavesdropperParameters',
    'NetworkParameters',
    'OutputParameters',

    # Presets and helpers
    'PresetConfigurations',
    'WavelengthBands',
    'ValidationRanges',
    'ErrorMessages',

    # Helper functions
    'db_to_linear',
    'linear_to_db',
    'transmission_to_db_loss',
]
