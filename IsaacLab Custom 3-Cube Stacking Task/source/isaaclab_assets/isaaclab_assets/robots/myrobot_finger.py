from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


def _resolve_fanuc_usd_path() -> str:
    env_path = os.environ.get("MY_FANUC_FINGER_USD")
    if env_path:
        return env_path

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "assets" / "my_fanuc_finger.usd"
        if candidate.is_file():
            return str(candidate)

    return "assets/my_fanuc_finger.usd"


MY_FANUC_FINGER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_resolve_fanuc_usd_path(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint_1": 0.60,
            "joint_2": -0.3,
            "joint_3": 0.56,
            "joint_4": -0.00,
            "joint_5": -0.08,
            "joint_6": -0.09,
            "panda_finger_joint[1-2]": 0.04,
        },
    ),
    actuators={
        "base": ImplicitActuatorCfg(
            joint_names_expr=["joint_1"],
            effort_limit_sim=80.0,
            stiffness=70.0,
            damping=12.0,
        ),
        "arm_big": ImplicitActuatorCfg(
            joint_names_expr=["joint_2"],
            effort_limit_sim=200.0,
            stiffness=90.0,
            damping=18.0,
        ),
        "hand_small": ImplicitActuatorCfg(
            joint_names_expr=["joint_[3-6]"],
            effort_limit_sim=80.0,
            stiffness=12.0,
            damping=70.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint[1-2]"],
            effort_limit_sim=200.0,
            stiffness=2000.0,
            damping=100.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
