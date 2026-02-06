"""Physical constants for Earth–Moon CR3BP (dimensional)."""

import numpy as np

G = 6.67430e-11                       # [m^3 / kg / s^2]
M_EARTH = 5.972168e24                 # [kg]
M_MOON = 7.346303e22                  # [kg]
D_EM = 3.84400e8                      # [m] Earth–Moon distance
R_EARTH = 6378137.0                   # [m]
R_MOON = 1737.4e3                     # [m]

MU = M_MOON / (M_EARTH + M_MOON)      # CR3BP mass parameter
MU_TOTAL = G * (M_EARTH + M_MOON)     # GM_total

L_STAR = D_EM                         # characteristic length
T_STAR = np.sqrt(L_STAR**3 / MU_TOTAL)  # characteristic time
V_STAR = L_STAR / T_STAR              # characteristic velocity
