# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_position_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    return torch.norm(des_pos_w - ee_pos_w, dim=1)


def ee_position_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    distance = ee_position_command_error(
        env,
        command_name=command_name,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
    )
    return 1 - torch.tanh(distance / std)


def ee_position_success_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.01,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    return (
        ee_position_command_error(
            env,
            command_name=command_name,
            robot_cfg=robot_cfg,
            ee_frame_cfg=ee_frame_cfg,
        )
        < threshold
    ).float()


def ee_orientation_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_w = quat_mul(robot.data.root_quat_w, command[:, 3:7])
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    return quat_error_magnitude(ee_quat_w, des_quat_w)


def gripper_hits_table_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[..., sensor_cfg.body_ids, :].norm(dim=-1)
    return (forces.max(dim=1)[0] > force_threshold).any(dim=1).float()


def ee_vertical_alignment(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    quat_w = ee_frame.data.target_quat_w[..., 0, :]
    z_unit = torch.zeros_like(quat_w[..., :3])
    z_unit[..., 2] = 1.0
    return torch.abs(math_utils.quat_apply(quat_w, z_unit)[:, 2])
