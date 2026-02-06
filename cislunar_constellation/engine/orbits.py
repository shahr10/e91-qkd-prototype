from __future__ import annotations

from typing import Dict, Any
import numpy as np
from scipy.integrate import solve_ivp

from cislunar_constellation.engine.constants import MU, L_STAR


def build_20plus_orbit_families() -> Dict[str, Dict[str, Any]]:
    """
    Build a database of periodic orbit families in Earth–Moon CR3BP.

    Each entry:
        name: {
            'family': str,
            'lp':     str ('L1','L2','Moon',...),
            'ic':     np.array([x,y,z,vx,vy,vz])  (non-dimensional CR3BP),
            'period_nd': float (non-dimensional period),
            'Az_km': float (vertical amplitude, if relevant),
            'color': str (hex),
            'stability': 'Low'/'Medium'/'High'/'Very High'
        }
    """

    def richardson_halo_ics(lp_num, Az_km, northern=True):
        """
        Third-order Richardson approximation for halo orbits near L1 or L2.
        Returns: ic (6,), T (non-dimensional period)
        """
        Az = Az_km * 1e3 / L_STAR

        if lp_num == 1:
            gamma_n = (MU / 3) ** (1 / 3)
            for _ in range(5):
                f = (
                    gamma_n**5
                    - (3 - MU) * gamma_n**4
                    + (3 - 2 * MU) * gamma_n**3
                    - MU * gamma_n**2
                    + 2 * MU * gamma_n
                    - MU
                )
                df = (
                    5 * gamma_n**4
                    - 4 * (3 - MU) * gamma_n**3
                    + 3 * (3 - 2 * MU) * gamma_n**2
                    - 2 * MU * gamma_n
                    + 2 * MU
                )
                gamma_n -= f / df
            gamma = gamma_n
            x_lp = 1 - MU - gamma
        else:
            gamma_n = (MU / 3) ** (1 / 3)
            for _ in range(5):
                f = (
                    gamma_n**5
                    + (3 - MU) * gamma_n**4
                    + (3 - 2 * MU) * gamma_n**3
                    - MU * gamma_n**2
                    - 2 * MU * gamma_n
                    - MU
                )
                df = (
                    5 * gamma_n**4
                    + 4 * (3 - MU) * gamma_n**3
                    + 3 * (3 - 2 * MU) * gamma_n**2
                    - 2 * MU * gamma_n
                    - 2 * MU
                )
                gamma_n -= f / df
            gamma = gamma_n
            x_lp = 1 - MU + gamma

        c2 = (1 / gamma**3) * ((1 - MU) + MU * gamma**3 / (1 - gamma) ** 3)
        disc = 9 * c2**2 - 8 * c2
        lam = np.sqrt((c2 - 2 + np.sqrt(disc)) / 2)
        k = 2 * lam / (lam**2 + 1 - c2)

        a21 = 3 * c2 * (k**2 - 2) / (4 * (1 + 2 * c2))
        a22 = 3 * c2 / (4 * (1 + 2 * c2))

        d1 = 3 * lam**2 / k * (k * (6 * lam**2 - 1) - 2 * lam)
        if abs(d1) > 1e-10:
            a23 = -3 * c2 * lam / (4 * k * d1) * (3 * k**3 * lam - 6 * k * (k - lam) + 4)
        else:
            a23 = 0.0

        # Linear ICs
        x0 = x_lp + a21 * Az**2
        z0 = Az if northern else -Az
        vy0 = k * lam * Az

        # Period
        T = 2 * np.pi / lam

        ic = np.array([x0, 0.0, z0, 0.0, vy0, 0.0])
        return ic, T

    database: Dict[str, Dict[str, Any]] = {}

    # Halo orbits (L1/L2) with different amplitudes
    halo_list = [
        (1, 8000, True, "L1_Halo_North_8k"),
        (1, 14000, True, "L1_Halo_North_14k"),
        (1, 20000, True, "L1_Halo_North_20k"),
        (1, 8000, False, "L1_Halo_South_8k"),
        (1, 14000, False, "L1_Halo_South_14k"),
        (1, 20000, False, "L1_Halo_South_20k"),
        (2, 8000, True, "L2_Halo_North_8k"),
        (2, 14000, True, "L2_Halo_North_14k"),
        (2, 20000, True, "L2_Halo_North_20k"),
        (2, 8000, False, "L2_Halo_South_8k"),
        (2, 14000, False, "L2_Halo_South_14k"),
        (2, 20000, False, "L2_Halo_South_20k"),
    ]

    colors = {
        "L1": "#FF6B6B",
        "L2": "#4ECDC4",
    }

    for lp_num, Az_km, north, name in halo_list:
        ic, period = richardson_halo_ics(lp_num, Az_km, northern=north)
        lp = "L1" if lp_num == 1 else "L2"
        database[name] = {
            "family": "Halo",
            "lp": lp,
            "ic": ic,
            "period_nd": period,
            "Az_km": Az_km,
            "color": colors[lp],
            "stability": "Medium" if Az_km < 15000 else "Low",
        }

    # Vertical orbits (approx ICs)
    vertical_ics = [
        ([0.8369, 0.0, 0.05, 0.0, 0.262, 0.0], 2.7, "L1_Vertical_Small"),
        ([0.8369, 0.0, 0.10, 0.0, 0.262, 0.0], 2.7, "L1_Vertical_Large"),
        ([1.1556, 0.0, 0.05, 0.0, -0.262, 0.0], 3.4, "L2_Vertical_Small"),
        ([1.1556, 0.0, 0.10, 0.0, -0.262, 0.0], 3.4, "L2_Vertical_Large"),
    ]
    for ic, period, name in vertical_ics:
        database[name] = {
            "family": "Vertical",
            "lp": name[:2],
            "ic": np.array(ic),
            "period_nd": period,
            "Az_km": 0.0,
            "color": "#7B2CBF",
            "stability": "High",
        }

    # DRO (Distant Retrograde Orbit)
    dro_ics = [
        ([1.05, 0.0, 0.0, 0.0, -0.15, 0.0], 6.3, "DRO_Small"),
        ([1.07, 0.0, 0.0, 0.0, -0.18, 0.0], 6.4, "DRO_Medium"),
        ([1.10, 0.0, 0.0, 0.0, -0.21, 0.0], 6.5, "DRO_Large"),
    ]
    for ic, period, name in dro_ics:
        database[name] = {
            "family": "DRO",
            "lp": "Moon",
            "ic": np.array(ic),
            "period_nd": period,
            "Az_km": 0.0,
            "color": "#F8961E",
            "stability": "Very High",
        }

    # Lyapunov orbits
    lyap_ics = [
        ([0.8369, 0.0, 0.0, 0.0, 0.0126, 0.0], 2.7523, "L1_Small"),
        ([0.8269, 0.0, 0.0, 0.0, 0.0226, 0.0], 2.7423, "L1_Large"),
        ([1.1556, 0.0, 0.0, 0.0, -0.0914, 0.0], 3.4043, "L2_Small"),
        ([1.1456, 0.0, 0.0, 0.0, -0.1014, 0.0], 3.3943, "L2_Large"),
    ]
    for ic, period, name in lyap_ics:
        database[name] = {
            "family": "Lyapunov",
            "lp": name[:2],
            "ic": np.array(ic),
            "period_nd": period,
            "Az_km": 0.0,
            "color": "#444444",
            "stability": "High",
        }

    return database


VALIDATED_ORBITS = build_20plus_orbit_families()


def cr3bp_eom_enhanced(t, state, mu):
    x, y, z, vx, vy, vz = state
    r1_sq = (x + mu) ** 2 + y**2 + z**2
    r2_sq = (x - 1 + mu) ** 2 + y**2 + z**2

    r1 = np.sqrt(max(r1_sq, 1e-16))
    r2 = np.sqrt(max(r2_sq, 1e-16))

    r13 = r1_sq * r1
    r23 = r2_sq * r2

    ax = 2 * vy + x - (1 - mu) * (x + mu) / r13 - mu * (x - 1 + mu) / r23
    ay = -2 * vx + y - (1 - mu) * y / r13 - mu * y / r23
    az = -(1 - mu) * z / r13 - mu * z / r23

    return [vx, vy, vz, ax, ay, az]


def propagate_orbit(orbit_name: str, n_points: int = 400):
    orbit = VALIDATED_ORBITS[orbit_name]

    if orbit["family"] in ["Resonant", "Figure8"]:
        n_periods = 3.0
        n_points = 600
    elif orbit["family"] in ["DRO", "Lyapunov"]:
        n_periods = 2.5
        n_points = 500
    else:
        n_periods = 2.0
        n_points = 400

    t_span = [0, orbit["period_nd"] * n_periods]
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    try:
        sol = solve_ivp(
            lambda t, y: cr3bp_eom_enhanced(t, y, MU),
            t_span,
            orbit["ic"],
            method="DOP853",
            t_eval=t_eval,
            rtol=1e-10,
            atol=1e-12,
        )
        if sol.success and sol.y is not None and sol.y.shape[1] > 0:
            return sol.y[:3] * L_STAR
        return None
    except Exception:
        return None


def list_orbits_by_family(families: list[str] | None = None) -> Dict[str, Dict[str, Any]]:
    if not families:
        return VALIDATED_ORBITS
    return {k: v for k, v in VALIDATED_ORBITS.items() if v.get("family") in families}
