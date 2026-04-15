# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp
import math 
##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """用于机器人与物体交互的举升场景配置。
    这是抽象基类实现，具体场景在派生类中定义，
    需设置目标物体、机器人及末端执行器坐标系。
    """

    # 机器人：将由代理环境配置填充
    robot: ArticulationCfg = MISSING
    # 末端执行器传感器：将由代理环境配置填充
    ee_frame: FrameTransformerCfg = MISSING
    # 目标对象：将由代理环境配置填充
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.8, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # 夹爪-桌面接触传感器（用于惩罚撞台）
    contact_gripper_table = ContactSensorCfg(
        # 覆盖 panda_hand 与 finger 链接
        prim_path="{ENV_REGEX_NS}/Robot/panda_.*",
        history_length=4,
        track_air_time=False,
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table.*"],
        debug_vis=False,
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """MDP的命令术语。"""

    # object_pose = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name=MISSING,  # will be set by agent env cfg
    #     resampling_time_range=(5.0, 5.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(0.7, 0.9), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
    #     ),
    # )

    target_pose = mdp.UniformPoseCommandCfg(
        class_type=mdp.FixedPoseCommand,
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        # 使用极大有限值，等价于“几乎不重采样”，避免 torch.uniform_ 处理 inf 报错
        resampling_time_range=(1e6, 1e6),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.6, 0.6), pos_y=(0.5, 0.5), pos_z=(0.05, 0.05), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )
    

@configclass
class ActionsCfg:
    """MDP的行动规范。"""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        cube_position = ObsTerm(
        func=mdp.object_position_in_robot_root_frame,
        
        params={"object_cfg": SceneEntityCfg("cube")},)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "target_pose"})

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_big_cube = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.05, 0.15), "y": (-0.35, -0.05), "z": (0.01, 0.01)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )

    reset_small_cube = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.05, 0.15), "y": (0.05, 0.35), "z": (0.01, 0.01)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube", body_names="Cube"),
        },
    )

    # 每次重置时将指令目标设为物体上方 5cm 处
    set_target_pose = EventTerm(
        func=mdp.set_target_pose_to_object,
        mode="reset",

        params={"command_name": "target_pose", "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.3}, weight=1.0)

    reaching_object_fine = RewTerm(func=mdp.object_ee_distance, params={"std": 0.15}, weight=1.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.03}, weight=1.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.08, "command_name": "target_pose"},
        weight=1.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.1,
            "minimal_height": 0.08,
            "command_name": "target_pose",
        },
        weight=8.0,
    )

    # 行动惩罚
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 夹爪撞台惩罚
    gripper_table_collision = RewTerm(
        func=mdp.gripper_hits_table_penalty,
        params={"sensor_cfg": SceneEntityCfg("contact_gripper_table"), "force_threshold": 1.0},
        weight=-5.0,
    )
    
    gripper_release = RewTerm(
        func=mdp.gripper_release_at_goal, # 使用上面的新函数
        params={
            "std": 0.05,             # 稍微宽松，引导它靠近
            "command_name": "target_pose",
            "gripper_joint_names": "panda_finger_joint[1-2]"
        },
        weight=20.0, 
    )

    gripper_closed_penalty = RewTerm(
        func=mdp.penalize_gripper_closed_at_goal,
        params={
            "command_name": "target_pose",

            "gripper_joint_names": "panda_finger_joint[1-2]",
            "open_threshold": 0.025,  # 单指 < 3cm 视为未松开
            "penalty": -1.0,
        },
        weight=1.0,  # 惩罚强度可根据需要调节
    )

    stack_success = RewTerm(
        func=mdp.success_at_goal_xy_static,
        params={
            "threshold": 0.02,
            "open_threshold": 0.032,
            "gripper_joint_names": "panda_finger_joint[1-2]",
        },
        weight=5000.0,  # 大奖
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )

    cube_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube")}
    )

    # # 在 TerminationsCfg 里添加
    # stack_success = DoneTerm(
    #     func=mdp.success_at_goal_xy_static,
    #     params={
    #         "threshold": 0.02,
    #         "open_threshold": 0.032,
    #         "gripper_joint_names": "panda_finger_joint[1-2]",
    #     },
    # )

@configclass
class CurriculumCfg:
    """MDP课程术语."""

    # action_rate_1 = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-3, "num_steps": 10000}
    # )

    # joint_vel_1 = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-3, "num_steps": 10000}
    # )

    # action_rate_2 = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 100000}
    # )

    # joint_vel_2 = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-2, "num_steps": 100000}
    # )

    # action_rate_3 = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 1000000}
    # )

    # joint_vel_3 = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 1000000}
    # )

    # # gripper_table_collision_1 = CurrTerm(
    # #     func=mdp.modify_reward_weight,
    # #     params={"term_name": "gripper_table_collision", "weight": -1.0, "num_steps":10000},
    # # )
    # # 中等惩罚
    # gripper_table_collision_2 = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "gripper_table_collision", "weight": -3.0, "num_steps": 100000},
    # )
    # # 目标惩罚
    # gripper_table_collision_3 = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "gripper_table_collision", "weight": -5.0, "num_steps": 1000000},
    # )

##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=3.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
