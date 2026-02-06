from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TimeConfig:
    horizon_hours: float = 24.0
    n_time: int = 36


@dataclass
class OrbitConfig:
    families: Optional[List[str]] = None
    max_candidates: int = 20
    orbit_names: Optional[List[str]] = None


@dataclass
class GroundStation:
    name: str
    lat: float
    lon: float
    alt: float


@dataclass
class GroundConfig:
    stations: List[GroundStation] = field(default_factory=lambda: [
        GroundStation("Hawaii", 19.82, -155.47, 4200),
        GroundStation("Tenerife", 28.30, -16.51, 2390),
        GroundStation("Ali", 32.33, 80.03, 5100),
        GroundStation("Canberra", -35.40, 149.13, 700),
        GroundStation("Chile", -30.24, -70.74, 2500),
    ])
    elevation_mask_deg: float = 10.0
    availability: float = 1.0


@dataclass
class LinkConfig:
    wavelength_m: float = 810e-9
    tx_aperture_m: float = 0.60
    tx_eff: float = 0.90
    pointing_jitter_rad: float = 0.2e-6
    rx_aperture_m: float = 8.0
    rx_eff: float = 0.75
    detector_eff: float = 0.95
    dark_rate_cps: float = 1.0
    rep_rate_hz: float = 5e9
    mu_sig: float = 0.6
    atm_loss_zenith_db: float = 1.5


@dataclass
class OptimizeConfig:
    ngen: int = 25
    pop_size: int = 50
    target_sats: int = 12
    cxpb: float = 0.7
    mutpb: float = 0.2
    seed: int = 42


@dataclass
class NetworkConfig:
    allow_isl: bool = True
    max_isl_partners: int = 3
    isl_distance_km: float = 150000.0


@dataclass
class OutputConfig:
    export_dir: str = "outputs"
    generate_figures: bool = False


@dataclass
class ConstellationConfig:
    time: TimeConfig = field(default_factory=TimeConfig)
    orbits: OrbitConfig = field(default_factory=OrbitConfig)
    ground: GroundConfig = field(default_factory=GroundConfig)
    link: LinkConfig = field(default_factory=LinkConfig)
    optimize: OptimizeConfig = field(default_factory=OptimizeConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> list[str]:
        warnings: list[str] = []

        if self.orbits.max_candidates <= 0:
            warnings.append("max_candidates should be > 0.")
        if self.optimize.target_sats <= 0:
            warnings.append("target_sats should be > 0.")
        if self.orbits.max_candidates < self.optimize.target_sats:
            warnings.append(
                "max_candidates is smaller than target_sats; optimization may be impossible."
            )
        if self.time.n_time < 6:
            warnings.append("n_time is very low; visibility estimates may be unstable.")
        if self.ground.elevation_mask_deg >= 60:
            warnings.append("High elevation mask may eliminate most links.")
        if self.ground.availability <= 0 or self.ground.availability > 1:
            warnings.append("availability should be in (0, 1].")

        return warnings
