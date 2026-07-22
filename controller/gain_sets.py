"""Named reactive-controller gain sets and their evidence status.

Only ``VERIFIED_BASELINE`` is allowed to provide controller defaults.  The
other sets preserve exploratory tuning without implying that independent
position/orientation sweep winners form a validated joint baseline.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GainSet:
    kp_pos: float
    kp_rot: float
    kd_pos: float
    kd_rot: float
    k_null: float
    damping: float

    def as_overrides(self):
        return {
            "KP_POS": self.kp_pos,
            "KP_ROT": self.kp_rot,
            "KD_POS": self.kd_pos,
            "KD_ROT": self.kd_rot,
            "K_NULL": self.k_null,
            "DAMPING": self.damping,
        }


# Last controller snapshot whose complete then-current suite passed: 8e4b6db.
VERIFIED_BASELINE = GainSet(
    kp_pos=2.0,
    kp_rot=2.0,
    kd_pos=0.3,
    kd_rot=0.3,
    k_null=1.0,
    damping=0.05,
)

# Preserved tuning history.  These are explicitly not canonical baselines.
EXPLORATORY_CANDIDATES = {
    "committed_20_0.9": GainSet(
        kp_pos=20.0,
        kp_rot=20.0,
        kd_pos=0.9,
        kd_rot=0.9,
        k_null=0.5,
        damping=0.05,
    ),
    "working_tree_32_2": GainSet(
        kp_pos=32.0,
        kp_rot=32.0,
        kd_pos=2.0,
        kd_rot=2.0,
        k_null=0.5,
        damping=0.05,
    ),
}
