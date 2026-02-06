from __future__ import annotations

from typing import Dict, Any

# Temporary bridge: reuse the extracted notebook code
from constellation_notebook import (
    VALIDATED_ORBITS,
    propagate_orbit,
)


def list_orbits_by_family(families: list[str] | None = None) -> Dict[str, Dict[str, Any]]:
    if not families:
        return VALIDATED_ORBITS
    return {k: v for k, v in VALIDATED_ORBITS.items() if v.get("family") in families}
