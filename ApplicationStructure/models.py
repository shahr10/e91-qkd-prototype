"""
================================================================================
DATA MODELS FOR E91 QKD
================================================================================

Data structures for E91 Quantum Key Distribution experiments.

This module contains:
- ExperimentConfig: Configuration dataclass for experiment parameters
- ExperimentResults: Results dataclass for storing simulation outputs
- MessageTest: Results from message encryption tests
- MessageListener: Background UDP listener for network communication

Author: Tyler Barr
Version: 7.0.0 Modular
Date: 2025

================================================================================
"""

import socket
import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from queue import Queue


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Comprehensive experiment configuration."""
    # Core
    backend: str = "qiskit"  # REQUIRED: Must use Qiskit or QuTiP
    num_pairs: int = 10000
    seed: int = 42
    # E91 protocol measurement angles
    # Alice: 0°, 22.5°, 45°  |  Bob: 22.5°, 45°, 67.5°
    # Theoretical CHSH: S = 2.414 (85% of Tsirelson bound 2.828)
    # With finite samples: S ≈ 2.40-2.41 (expected and correct)
    alice_angles: List[float] = field(default_factory=lambda: [0.0, 0.39269908169872414, 0.7853981633974483])
    bob_angles: List[float] = field(default_factory=lambda: [0.39269908169872414, 0.7853981633974483, 1.1780972450961724])
    alice_basis_probs: Optional[List[float]] = None
    bob_basis_probs: Optional[List[float]] = None
    basis_mapping: Optional[List[Tuple[int, int, str]]] = None

    # Security
    epsilon_sec: float = 1e-9
    epsilon_cor: float = 1e-15
    pe_fraction: float = 0.1
    chsh_fraction: float = 0.1  # Phase 5B: Fraction of detected events for CHSH test
    # CHSH test indices (CRITICAL: defines which settings compute CHSH statistic)
    # CHSH = |E(a0,b0) - E(a0,b1) + E(a1,b0) + E(a1,b1)|
    # For Tsirelson-optimal: use basis indices {0,2} for both Alice and Bob
    chsh_a0_idx: int = 0  # Alice's first CHSH setting
    chsh_a1_idx: int = 2  # Alice's second CHSH setting
    chsh_b0_idx: int = 0  # Bob's first CHSH setting
    chsh_b1_idx: int = 2  # Bob's second CHSH setting
    ir_protocol: str = "BBBSS"
    f_EC: float = 1.2
    block_length: int = 1024
    code_rate: float = 0.5

    # Device-Independent
    di_bound_type: str = "default_di_bound"
    enable_routed_di: bool = False
    routing_efficiency_f: float = 0.95
    routing_efficiency_d: float = 0.90

    # Logical Noise
    enable_depolarizing_noise: bool = False
    depolarizing_alpha: float = 0.06
    enable_visibility_reduction: bool = False
    visibility: float = 0.98
    enable_intrinsic_error: bool = False
    intrinsic_error_ed: float = 0.01

    # Channel Loss
    enable_fiber_loss: bool = False
    distance_km_A: float = 5.0
    distance_km_B: float = 5.0
    loss_dB_A: float = 0.0
    loss_dB_B: float = 0.0
    fiber_loss_dB_per_km: float = 0.2
    enable_insertion_loss: bool = False
    insertion_loss_dB: float = 0.5
    enable_misalignment: bool = False
    polarization_drift_deg: float = 2.0
    # Repeaters (effective loss compensation model)
    enable_repeaters: bool = False
    num_repeaters: int = 0
    repeater_gain_dB: float = 0.0

    # Detectors
    enable_detector_loss: bool = False
    heralding_efficiency: float = 0.9  # Backward compatibility
    heralding_efficiency_A: float = 0.9  # Alice-side heralding
    heralding_efficiency_B: float = 0.9  # Bob-side heralding
    end_detector_efficiency: float = 0.8  # Backward compatibility
    end_detector_efficiency_A: float = 0.8  # Alice-side detector QE
    end_detector_efficiency_B: float = 0.8  # Bob-side detector QE
    enable_dark_counts: bool = False
    dark_cps: float = 100.0
    dark_prob: float = 0.001
    use_dark_cps: bool = False
    enable_timing_jitter: bool = False
    jitter_ns: float = 0.5  # Backward compatibility (use jitter_A_ns/jitter_B_ns for per-side)
    jitter_A_ns: float = 0.5  # Alice-side timing jitter
    jitter_B_ns: float = 0.5  # Bob-side timing jitter
    enable_deadtime: bool = False
    deadtime_ns: float = 50.0  # Backward compatibility
    deadtime_A_ns: float = 50.0  # Alice-side detector deadtime
    deadtime_B_ns: float = 50.0  # Bob-side detector deadtime
    enable_afterpulsing: bool = False
    afterpulsing_prob: float = 0.05  # Initial afterpulse probability P0
    afterpulsing_tau_ns: float = 100.0  # Trap lifetime (exponential decay constant)
    afterpulsing_delay_ns: float = 100.0  # Backward compatibility (use afterpulsing_tau_ns)
    enable_saturation: bool = False
    saturation_rate: float = 1e6
    # Double-click handling (2 detectors per side: D0_A, D1_A, D0_B, D1_B)
    enable_double_click_squashing: bool = False
    squashing_model: str = "random"  # "random" or "discard"

    # Background
    enable_background: bool = False
    Y0: float = 1e-6
    background_cps: float = 500.0
    # Timing windows (FIX C2: Separated gate and coincidence windows)
    gate_window_ns: float = 1.0  # Detection gate width (for noise click probabilities)
    coincidence_window_ns: float = 1.0  # Time window for coincidence pairing
    enable_time_tagging: bool = False
    repetition_rate_Hz: float = 1e6

    # Multi-pair SPDC (approximate)
    # When enabled, the simulator samples a pair number k per trial from an SPDC distribution
    # and models threshold-detector double-click behavior via squashing (random/discard).
    enable_multi_pair: bool = False
    spdc_brightness_mu: float = 0.1  # Mean pairs per pulse (μ)
    spdc_distribution: str = "thermal"  # "thermal" (single-mode) or "poisson" (many-mode)
    pair_rate: float = 1e6
    pump_power_mW: float = 10.0
    wavelength_nm: float = 810.0
    filter_bandwidth_nm: float = 3.0

    # Satellite
    enable_satellite: bool = False
    beam_divergence_urad: float = 10.0
    pointing_jitter_urad: float = 5.0
    receiver_fov_urad: float = 100.0
    slant_range_km: float = 500.0
    transmitter_aperture_m: float = 0.3
    receiver_aperture_m: float = 1.0
    is_daytime: bool = False
    satellite_background_cps: float = 1000.0
    enable_turbulence: bool = False
    cn2: float = 1e-15

    # Adversary (Eve)
    enable_eavesdropper: bool = False
    eve_model: str = "intercept_resend"
    eve_intercept_prob: float = 0.0

    # Output
    output_directory: str = "./results"
    enable_plots: bool = True
    plot_format: str = "png"
    enable_structured_export: bool = False
    enable_bb84_comparison: bool = False
    enable_tomography: bool = False
    preset_name: str = "Custom"
    # State preparation
    bell_state: str = "phi_plus"

    def __post_init__(self):
        """
        Initialize basis mapping and normalize shared/per-side parameters.

        CRITICAL FIX: UI and presets set "shared" parameters (e.g., heralding_efficiency),
        but the simulator uses per-side A/B parameters (e.g., heralding_efficiency_A/B).
        This normalization ensures shared values propagate to A/B when they differ from defaults.
        """
        # Initialize basis mapping if not provided
        if self.basis_mapping is None:
            self.basis_mapping = []
            for i in range(3):
                for j in range(3):
                    # Key: use matching angles - Alice[1]=Bob[0]=π/8 and Alice[2]=Bob[1]=π/4
                    purpose = "key" if (i == 1 and j == 0) or (i == 2 and j == 1) else "chsh"
                    self.basis_mapping.append((i, j, purpose))

        # PARAMETER NORMALIZATION: Propagate shared → A/B when A/B are still at defaults
        # This fixes the critical bug where UI/preset values are silently ignored

        # Detector efficiencies (0.8 is the default for end detector)
        if (self.end_detector_efficiency_A == 0.8 and
            self.end_detector_efficiency_B == 0.8 and
            self.end_detector_efficiency != 0.8):
            self.end_detector_efficiency_A = self.end_detector_efficiency
            self.end_detector_efficiency_B = self.end_detector_efficiency

        # Heralding efficiencies (0.9 is the default)
        if (self.heralding_efficiency_A == 0.9 and
            self.heralding_efficiency_B == 0.9 and
            self.heralding_efficiency != 0.9):
            self.heralding_efficiency_A = self.heralding_efficiency
            self.heralding_efficiency_B = self.heralding_efficiency

        # Deadtime (50.0 ns is the default)
        if (self.deadtime_A_ns == 50.0 and
            self.deadtime_B_ns == 50.0 and
            self.deadtime_ns != 50.0):
            self.deadtime_A_ns = self.deadtime_ns
            self.deadtime_B_ns = self.deadtime_ns

        # Timing jitter (0.5 ns is the default)
        if (self.jitter_A_ns == 0.5 and
            self.jitter_B_ns == 0.5 and
            self.jitter_ns != 0.5):
            self.jitter_A_ns = self.jitter_ns
            self.jitter_B_ns = self.jitter_ns


# ============================================================================
# EXPERIMENT RESULTS
# ============================================================================

@dataclass
class ExperimentResults:
    """
    Comprehensive experiment results.

    KEY-RATE UNITS (important for comparing results):
    - secret_fraction_asymptotic: secret bits per sifted bit [dimensionless]
    - secret_fraction_finite: secret bits per sifted bit [dimensionless]
    - key_rate_asymptotic: secret bits per detected pair = secret_fraction × sifting_efficiency
    - key_rate_finite: secret bits per detected pair = secret_fraction × sifting_efficiency

    To compute other common metrics:
    - Secret bits per emitted pair: key_rate × detection_efficiency
    - Secret bits per second: key_rate × detection_efficiency × repetition_rate_Hz
    """
    num_pairs_generated: int
    num_pairs_detected: int
    detection_efficiency: float
    num_sifted: int
    sifting_efficiency: float
    qber: float
    chsh_S: float
    num_key_bits: int
    secret_fraction_asymptotic: float  # bits per sifted bit
    secret_fraction_finite: float  # bits per sifted bit
    key_rate_asymptotic: float  # bits per detected pair
    key_rate_finite: float  # bits per detected pair
    correlators: Dict[str, float]
    routed_di_witness: Optional[float] = None
    combined_efficiency: Optional[float] = None
    avg_coincidence_delay_ns: Optional[float] = None
    coincidence_accidental_ratio: Optional[float] = None
    multi_pair_events: Optional[int] = None
    per_basis_qber: Optional[Dict[str, float]] = None  # Phase 5B: QBER for each basis pair
    # Singles and accidentals (MANDATORY for experimental validation)
    singles_rate_alice_Hz: Optional[float] = None  # Alice singles rate (Hz)
    singles_rate_bob_Hz: Optional[float] = None  # Bob singles rate (Hz)
    coincidence_rate_Hz: Optional[float] = None  # Coincidence rate (Hz)
    accidentals_estimate: Optional[float] = None  # Expected accidentals count
    alice_results: List[int] = field(default_factory=list)
    bob_results: List[int] = field(default_factory=list)
    sifted_alice: List[int] = field(default_factory=list)
    sifted_bob: List[int] = field(default_factory=list)
    time_tags_alice: Optional[List[float]] = None
    time_tags_bob: Optional[List[float]] = None
    execution_time: float = 0.0
    eve_intercepted: Optional[int] = None


# ============================================================================
# MESSAGE TEST RESULTS
# ============================================================================

@dataclass
class MessageTest:
    """Results from a message transmission test."""
    mode: str
    original_message: str
    encrypted_message: bytes
    decrypted_message: str
    key_used: List[int]
    key_length_bits: int
    encryption_method: str
    success: bool
    error_message: Optional[str] = None
    transmission_time: float = 0.0
    alice_key: Optional[List[int]] = None
    bob_key: Optional[List[int]] = None
    key_mismatch_positions: Optional[List[int]] = None


# ============================================================================
# MESSAGE LISTENER (UDP NETWORKING)
# ============================================================================

@dataclass
class MessageListener:
    """Background UDP message listener using threading."""
    active: bool = False
    port: int = 8765
    received_messages: List[Dict] = field(default_factory=list)
    _thread: Optional[threading.Thread] = None
    _stop_flag: threading.Event = field(default_factory=threading.Event)
    _message_queue: Queue = field(default_factory=Queue)

    def start(self, port: int):
        """Start background listening thread."""
        if self.active:
            return
        self.port = port
        self.active = True
        self._stop_flag.clear()

        # Test if port is available before starting thread
        try:
            test_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            test_sock.bind(("0.0.0.0", self.port))
            test_sock.close()
        except Exception as e:
            self.active = False
            raise RuntimeError(f"Port {port} is already in use or unavailable: {e}")

        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background listener."""
        self.active = False
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        # Reset the thread reference
        self._thread = None

    def _listen_loop(self):
        """Background thread that continuously listens for UDP messages."""
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Set SO_REUSEADDR to allow port reuse
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", self.port))
            sock.settimeout(0.5)  # Short timeout to check stop flag regularly
            logging.info(f"UDP listener started on port {self.port}")

            while not self._stop_flag.is_set() and self.active:
                try:
                    data, addr = sock.recvfrom(65535)
                    message_info = {
                        'data': data,
                        'sender': f"{addr[0]}:{addr[1]}",
                        'timestamp': time.time(),
                        'size': len(data)
                    }
                    self._message_queue.put(message_info)
                    self.received_messages.append(message_info)
                    logging.info(f"Received {len(data)} bytes from {addr[0]}:{addr[1]}")
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.active:  # Only log if we're still supposed to be active
                        logging.warning(f"Listener error: {e}")
                    break
        except Exception as e:
            logging.error(f"Failed to start listener on port {self.port}: {e}")
            self.active = False
        finally:
            if sock:
                sock.close()
            logging.info(f"UDP listener stopped on port {self.port}")

    def get_new_messages(self) -> List[Dict]:
        """Get all new messages from queue."""
        messages = []
        while not self._message_queue.empty():
            try:
                messages.append(self._message_queue.get_nowait())
            except:
                break
        return messages

    def clear_messages(self):
        """Clear all received messages."""
        self.received_messages.clear()
        while not self._message_queue.empty():
            try:
                self._message_queue.get_nowait()
            except:
                break


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'ExperimentConfig',
    'ExperimentResults',
    'MessageTest',
    'MessageListener',
]
