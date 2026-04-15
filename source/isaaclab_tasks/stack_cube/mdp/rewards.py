# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("cube")
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object_ee_distance = torch.norm(object.data.root_pos_w - ee_frame.data.target_pos_w[..., 0, :], dim=1)
    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def gripper_release_at_goal(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    gripper_joint_names: str = "panda_finger_joint[1-2]",
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    object = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    distance_xy = torch.norm(des_pos_w[:, :2] - object.data.root_pos_w[:, :2], dim=1)

    gripper_joint_ids, _ = robot.find_joints(gripper_joint_names)
    avg_gripper_pos = torch.mean(robot.data.joint_pos[:, gripper_joint_ids], dim=1)
    gripper_openness = torch.clamp((avg_gripper_pos - 0.02) / 0.02, 0.0, 1.0)
    return (1 - torch.tanh(distance_xy / std)) * gripper_openness


def penalize_gripper_closed_at_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    goal_std: float = 0.05,
    minimal_height: float = 0.08,
    gripper_joint_names: str = "panda_finger_joint[1-2]",
    open_threshold: float = 0.032,
    penalty: float = -1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    cmd = env.command_manager.get_command(command_name)

    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, cmd[:, :3])
    dist = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)
    proximity = 1 - torch.tanh(dist / goal_std)
    lifted = (obj.data.root_pos_w[:, 2] > minimal_height).float()

    joint_ids, _ = robot.find_joints(gripper_joint_names)
    joint_pos = robot.data.joint_pos[:, joint_ids].mean(dim=1)
    return penalty * proximity * lifted * (joint_pos < open_threshold).float()


def gripper_hits_table_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[..., sensor_cfg.body_ids, :].norm(dim=-1)
    return (forces.max(dim=1)[0] > force_threshold).any(dim=1).float()


def success_at_goal_xy_static(
    env: ManagerBasedRLEnv,
    command_name: str = "target_pose",
    threshold: float = 0.02,
    velocity_threshold: float = 0.05,
    open_threshold: float = 0.03,
    gripper_joint_names: str = "panda_finger_joint[1-2]",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    object = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    distance_xy = torch.norm(des_pos_w[:, :2] - object.data.root_pos_w[:, :2], dim=1)

    gripper_joint_ids, _ = robot.find_joints(gripper_joint_names)
    gripper_pos = robot.data.joint_pos[:, gripper_joint_ids]
    object_lin_vel = torch.norm(object.data.root_vel_w[:, :3], dim=1)

    return (
        (distance_xy < threshold)
        & (torch.mean(gripper_pos, dim=1) > open_threshold)
        & (object_lin_vel < velocity_threshold)
    )
