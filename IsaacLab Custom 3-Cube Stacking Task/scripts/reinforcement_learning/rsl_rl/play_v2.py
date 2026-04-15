# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import math
import os
import sys
import time

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play two-block stack with one reach policy and BT control.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# BT / 技能参数
parser.add_argument("--reach_threshold", type=float, default=0.013, help="末端距离目标点小于这个值时，认为已经到达。")

parser.add_argument("--grasp_reach_threshold", type=float, default=0.013, help="抓取阶段的 reached 阈值。")
parser.add_argument("--place_reach_threshold", type=float, default=0.013, help="放置阶段的 reached 阈值。")
parser.add_argument("--retreat_reach_threshold", type=float, default=0.04, help="撤手阶段的 reached 阈值。")


parser.add_argument("--pre_grasp_offset", type=float, default=0.0, help="抓方块前，先停在方块正上方多高的位置。")
parser.add_argument("--pre_place_offset", type=float, default=0.02, help="放方块前，先停在目标位置正上方多高的位置。")
parser.add_argument("--lift_height", type=float, default=0.15, help="抓住方块后，抬升阶段要把末端抬到的目标高度。")
parser.add_argument("--grasp_steps", type=int, default=20, help="夹爪闭合后，额外保持闭合多少个仿真步。")
parser.add_argument("--release_steps", type=int, default=30, help="夹爪张开后，额外保持张开多少个仿真步。")
parser.add_argument("--stage_timeout_s", type=float, default=5, help="某个阶段卡住超过多少秒，就直接重置环境。")
parser.add_argument("--final_hold_s", type=float, default=3.0, help="全部完成后，机械臂保持当前状态多少秒再重置。")
parser.add_argument("--drop_z", type=float, default=0.0, help="方块高度低于这个值时，认为掉落并重置环境。")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab.utils.math as math_utils
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.stack_3cube_v2 import mdp
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401

STAGE_PRE_GRASP_C1 = 0
STAGE_GRASP_C1 = 1
STAGE_LIFT_C1 = 2
STAGE_PRE_PLACE_C1 = 3
STAGE_RELEASE_C1 = 4
STAGE_RETREAT_1 = 5
STAGE_PRE_GRASP_C2 = 6
STAGE_GRASP_C2 = 7
STAGE_LIFT_C2 = 8
STAGE_PRE_PLACE_C2 = 9
STAGE_RELEASE_C2 = 10
STAGE_RETREAT_2 = 11
STAGE_FINAL_HOLD = 12

STAGE_NAMES = {
    STAGE_PRE_GRASP_C1: "pre_grasp_c1",
    STAGE_GRASP_C1: "grasp_c1",
    STAGE_LIFT_C1: "lift_c1",
    STAGE_PRE_PLACE_C1: "pre_place_c1",
    STAGE_RELEASE_C1: "release_c1",
    STAGE_RETREAT_1: "retreat_1",
    STAGE_PRE_GRASP_C2: "pre_grasp_c2",
    STAGE_GRASP_C2: "grasp_c2",
    STAGE_LIFT_C2: "lift_c2",
    STAGE_PRE_PLACE_C2: "pre_place_c2",
    STAGE_RELEASE_C2: "release_c2",
    STAGE_RETREAT_2: "retreat_2",
    STAGE_FINAL_HOLD: "final_hold",
}

GRIPPER_OPEN_POS = 0.04
GRIPPER_CLOSE_POS = 0.022
GRIPPER_JOINT_NAMES = "panda_finger_joint[1-2]"

# retreat 也走 target_pose，而不是再用脚本直接写关节。
# 这样整个低层技能始终只有“到目标位姿”这一件事。
RETREAT_POS_B = (0.60, 0.00, 0.18)
RETREAT_DELTA_Z = 0.10
PLACE_HEIGHT = 0.05
PLACE_HEIGHT_TOL = 0.02


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 该环境默认会在 EE 到达 target_pose 时直接 done。
    # 这里关闭该终止项，把“到达目标”的判定交给 BT 自己处理。
    if hasattr(env_cfg.terminations, "ee_position_success"):
        env_cfg.terminations.ee_position_success = None

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    if resume_path is None:
        raise FileNotFoundError("No reach checkpoint found. Pass --checkpoint or configure load_run/load_checkpoint.")

    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(env_cfg.log_dir, "videos", "play_BT_reach"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    print(f"[INFO] Loading reach checkpoint: {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    robot = env.unwrapped.scene["robot"]
    ee_frame = env.unwrapped.scene["ee_frame"]
    command_term = env.unwrapped.command_manager.get_term("target_pose")
    finger_joint_ids, _ = robot.find_joints(GRIPPER_JOINT_NAMES)

    stage = torch.full((env.num_envs,), STAGE_PRE_GRASP_C1, dtype=torch.long, device=env.device)
    stage_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    hold_steps_left = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    gripper_steps_left = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    retreat_target_1_b = torch.zeros((env.num_envs, 3), dtype=torch.float, device=env.device)
    retreat_target_2_b = torch.zeros((env.num_envs, 3), dtype=torch.float, device=env.device)

    dt = env.unwrapped.step_dt
    stage_timeout_steps = max(1, int(args_cli.stage_timeout_s / dt))
    final_hold_steps = max(1, int(args_cli.final_hold_s / dt))

    roll = torch.full((env.num_envs,), math.pi, device=env.device)
    zero = torch.zeros_like(roll)
    fixed_quat_b = math_utils.quat_from_euler_xyz(roll, zero, zero)

    def log_stage(mask: torch.Tensor, stage_id: int, reason: str):
        ids = torch.where(mask)[0]
        if ids.numel() == 0:
            return
        ids_preview = ids[:8].tolist()
        suffix = "..." if ids.numel() > 8 else ""
        print(f"[STAGE] {reason}: {STAGE_NAMES[stage_id]} envs={ids_preview}{suffix}")

    def reset_bt_state(mask: torch.Tensor):
        stage[mask] = STAGE_PRE_GRASP_C1
        stage_steps[mask] = 0
        hold_steps_left[mask] = 0
        gripper_steps_left[mask] = 0
        retreat_target_1_b[mask, 0] = RETREAT_POS_B[0]
        retreat_target_1_b[mask, 1] = RETREAT_POS_B[1]
        retreat_target_1_b[mask, 2] = RETREAT_POS_B[2]
        retreat_target_2_b[mask, 0] = RETREAT_POS_B[0]
        retreat_target_2_b[mask, 1] = RETREAT_POS_B[1]
        retreat_target_2_b[mask, 2] = RETREAT_POS_B[2]

    def reset_envs(mask: torch.Tensor, reason: str):
        ids = torch.where(mask)[0]
        if ids.numel() == 0:
            return
        env.unwrapped.reset(env_ids=ids)
        reset_bt_state(mask)
        policy_nn.reset(mask.to(dtype=torch.long))
        log_stage(mask, STAGE_PRE_GRASP_C1, reason)

    def world_to_base(pos_w: torch.Tensor) -> torch.Tensor:
        base_pos_w = robot.data.root_pos_w
        base_quat_inv = math_utils.quat_inv(robot.data.root_quat_w)
        return math_utils.quat_apply(base_quat_inv, pos_w - base_pos_w)

    def asset_pos_b(asset_name: str) -> torch.Tensor:
        asset = env.unwrapped.scene[asset_name]
        return world_to_base(asset.data.root_pos_w)

    def set_target_pose_b(mask: torch.Tensor, pos_b: torch.Tensor):
        ids = torch.where(mask)[0]
        if ids.numel() == 0:
            return
        command_term.pose_command_b[ids, :3] = pos_b[ids]
        command_term.pose_command_b[ids, 3:] = fixed_quat_b[ids]

    def capture_retreat_target(mask: torch.Tensor, target_buffer_b: torch.Tensor):
        ids = torch.where(mask)[0]
        if ids.numel() == 0:
            return
        ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
        ee_pos_b = world_to_base(ee_pos_w)
        target_buffer_b[ids] = ee_pos_b[ids]
        target_buffer_b[ids, 2] += RETREAT_DELTA_Z

    def apply_bt_targets():
        """按当前阶段把目标位姿写入 target_pose。

        这里的 target_pose 就是 reach policy 唯一会跟踪的低层命令。
        """

        cube1_pos_b = asset_pos_b("cube1")
        cube2_pos_b = asset_pos_b("cube2")
        object_pos_b = asset_pos_b("object")

        # 抓取 cube1：先对准 cube1 上方一个很小的偏置。
        target_pre_grasp_c1 = cube1_pos_b.clone()
        target_pre_grasp_c1[:, 2] += args_cli.pre_grasp_offset

        # 提升 cube1：保持当前抓取区域的 XY，只把 z 拉到安全高度。
        target_lift_c1 = cube1_pos_b.clone()
        target_lift_c1[:, 2] = args_cli.lift_height

        # 放置 cube1：目标是 object 上方 0.05，再给一点预放置高度。
        target_pre_place_c1 = object_pos_b.clone()
        target_pre_place_c1[:, 2] += PLACE_HEIGHT + args_cli.pre_place_offset

        # 抓取 cube2
        target_pre_grasp_c2 = cube2_pos_b.clone()
        target_pre_grasp_c2[:, 2] += args_cli.pre_grasp_offset

        # 提升 cube2
        target_lift_c2 = cube2_pos_b.clone()
        target_lift_c2[:, 2] = args_cli.lift_height

        # 放置 cube2：目标是 cube1 上方 0.05，再给一点预放置高度。
        # target_pre_place_c2 = cube1_pos_b.clone()
        # target_pre_place_c2[:, 2] += PLACE_HEIGHT + args_cli.pre_place_offset

        target_pre_place_c2 = object_pos_b.clone()
        target_pre_place_c2[:, 2] += 0.11 + args_cli.pre_place_offset

        set_target_pose_b((stage == STAGE_PRE_GRASP_C1) | (stage == STAGE_GRASP_C1), target_pre_grasp_c1)
        set_target_pose_b(stage == STAGE_LIFT_C1, target_lift_c1)
        set_target_pose_b((stage == STAGE_PRE_PLACE_C1) | (stage == STAGE_RELEASE_C1), target_pre_place_c1)
        set_target_pose_b(stage == STAGE_RETREAT_1, retreat_target_1_b)

        set_target_pose_b((stage == STAGE_PRE_GRASP_C2) | (stage == STAGE_GRASP_C2), target_pre_grasp_c2)
        set_target_pose_b(stage == STAGE_LIFT_C2, target_lift_c2)
        set_target_pose_b((stage == STAGE_PRE_PLACE_C2) | (stage == STAGE_RELEASE_C2), target_pre_place_c2)
        set_target_pose_b((stage == STAGE_RETREAT_2) | (stage == STAGE_FINAL_HOLD), retreat_target_2_b)

    def set_gripper_targets():
        # 只有抓取/搬运阶段闭爪，其余阶段全部张开。
        desired = torch.full((env.num_envs, len(finger_joint_ids)), GRIPPER_OPEN_POS, device=env.device)
        closed_mask = (
            (stage == STAGE_GRASP_C1)
            | (stage == STAGE_LIFT_C1)
            | (stage == STAGE_PRE_PLACE_C1)
            | ((stage == STAGE_RELEASE_C1) & (gripper_steps_left > 0))
            | (stage == STAGE_GRASP_C2)
            | (stage == STAGE_LIFT_C2)
            | (stage == STAGE_PRE_PLACE_C2)
            | ((stage == STAGE_RELEASE_C2) & (gripper_steps_left > 0))
        )
        desired[closed_mask] = GRIPPER_CLOSE_POS
        robot.set_joint_position_target(desired, joint_ids=finger_joint_ids)

    # def ee_reached() -> torch.Tensor:
    #     return mdp.ee_reached_command_position(
    #         env.unwrapped,
    #         command_name="target_pose",
    #         threshold=args_cli.reach_threshold,
    #         ee_frame_cfg=SceneEntityCfg("ee_frame"),
    #     )

    def ee_reached(threshold: float) -> torch.Tensor:
        return mdp.ee_reached_command_position(
            env.unwrapped,
            command_name="target_pose",
            threshold=threshold,
            ee_frame_cfg=SceneEntityCfg("ee_frame"),
        )


    def object_xy_aligned(top_name: str, base_name: str, threshold: float = 0.025) -> torch.Tensor:
        top = env.unwrapped.scene[top_name]
        base = env.unwrapped.scene[base_name]
        xy_dist = torch.norm(top.data.root_pos_w[:, :2] - base.data.root_pos_w[:, :2], dim=1)
        return xy_dist < threshold


    def ee_slow(vel_threshold: float = 0.08) -> torch.Tensor:
        hand_body_id = robot.find_bodies("panda_hand")[0][0]
        hand_vel_w = robot.data.body_lin_vel_w[:, hand_body_id, :]
        return torch.norm(hand_vel_w, dim=1) < vel_threshold


    def object_lifted(object_name: str, minimal_height: float = 0.08) -> torch.Tensor:
        obj = env.unwrapped.scene[object_name]
        return obj.data.root_pos_w[:, 2] > minimal_height

    def object_dropped(object_name: str) -> torch.Tensor:
        obj = env.unwrapped.scene[object_name]
        return obj.data.root_pos_w[:, 2] < args_cli.drop_z

    def is_gripper_open(open_threshold: float = 0.03) -> torch.Tensor:
        gripper_pos = robot.data.joint_pos[:, finger_joint_ids]
        return torch.mean(gripper_pos, dim=1) > open_threshold

    def stacked_success(top_name: str, base_name: str) -> torch.Tensor:
        """简化的堆叠成功判据。

        条件：
        1. XY 对齐
        2. 两个块的中心高度差接近一个方块高度
        3. 顶块速度够小，说明已经放稳
        4. 夹爪已经张开，避免“夹着也算成功”
        """
        top = env.unwrapped.scene[top_name]
        base = env.unwrapped.scene[base_name]
        xy_dist = torch.norm(top.data.root_pos_w[:, :2] - base.data.root_pos_w[:, :2], dim=1)
        z_gap = top.data.root_pos_w[:, 2] - base.data.root_pos_w[:, 2]
        lin_vel = torch.norm(top.data.root_vel_w[:, :3], dim=1)
        return (
            (xy_dist < 0.025)
            & (z_gap > (PLACE_HEIGHT - PLACE_HEIGHT_TOL))
            & (z_gap < (PLACE_HEIGHT + PLACE_HEIGHT_TOL))
            & (lin_vel < 0.05)
        )

    apply_bt_targets()
    obs = env.get_observations()
    timestep = 0
    log_stage(torch.ones(env.num_envs, dtype=torch.bool, device=env.device), STAGE_PRE_GRASP_C1, "init")

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            # ---------- 1) 先让 BT 写入当前阶段的 target_pose ----------
            apply_bt_targets()

            # ---------- 2) 读取当前阶段信号 ----------
            reached = ee_reached(args_cli.grasp_reach_threshold)

            reached_retreat = ee_reached(args_cli.retreat_reach_threshold)

            cube1_lifted = object_lifted("cube1")
            cube2_lifted = object_lifted("cube2")
            cube1_stacked = stacked_success("cube1", "object")
            cube2_stacked = stacked_success("cube2", "cube1")

            place1_aligned = object_xy_aligned("cube1", "object")
            place2_aligned = object_xy_aligned("cube2", "cube1")

            drop_any = object_dropped("cube1") | object_dropped("cube2")

            # 如果已经进入第二块阶段，但第一块掉了，直接整回合重来。
            c1_invalid = (stage >= STAGE_PRE_GRASP_C2) & (~cube1_stacked)
            # if c1_invalid.any():
            #     reset_envs(c1_invalid, "c1 invalid----------------------------------")
            #     obs = env.get_observations()
            #     continue

            timeout_mask = (stage != STAGE_FINAL_HOLD) & (stage_steps > stage_timeout_steps)
            force_reset = drop_any | timeout_mask
            if force_reset.any():
                reset_envs(force_reset, "drop/timeout----------------------------------")
                obs = env.get_observations()
                continue

            # ---------- 3) 阶段切换 ----------
            # 用本轮开始时的阶段快照做判定，避免一次循环内连续跨多个 stage。
            stage_curr = stage.clone()

            # cube1 抓取链路
            ee_is_slow = ee_slow()
            
            enter_grasp_c1 = (stage_curr == STAGE_PRE_GRASP_C1) & reached
            if enter_grasp_c1.any():
                stage[enter_grasp_c1] = STAGE_GRASP_C1
                stage_steps[enter_grasp_c1] = 0
                gripper_steps_left[enter_grasp_c1] = args_cli.grasp_steps
                log_stage(enter_grasp_c1, STAGE_GRASP_C1, "reach cube1")

            finish_grasp_c1 = (stage_curr == STAGE_GRASP_C1) & (gripper_steps_left <= 0)
            if finish_grasp_c1.any():
                stage[finish_grasp_c1] = STAGE_LIFT_C1
                stage_steps[finish_grasp_c1] = 0
                log_stage(finish_grasp_c1, STAGE_LIFT_C1, "close gripper cube1")

            finish_lift_c1 = (stage_curr == STAGE_LIFT_C1) & cube1_lifted & (stage_steps >= 20)
            if finish_lift_c1.any():
                stage[finish_lift_c1] = STAGE_PRE_PLACE_C1
                stage_steps[finish_lift_c1] = 20
                log_stage(finish_lift_c1, STAGE_PRE_PLACE_C1, "cube1 lifted")

            enter_release_c1 = (stage_curr == STAGE_PRE_PLACE_C1) & place1_aligned & ee_is_slow
            # enter_release_c1 = (stage == STAGE_PRE_PLACE_C1) & reached 
            if enter_release_c1.any():
                stage[enter_release_c1] = STAGE_RELEASE_C1
                stage_steps[enter_release_c1] = 0
                gripper_steps_left[enter_release_c1] = args_cli.release_steps
                log_stage(enter_release_c1, STAGE_RELEASE_C1, "arrive place1")

            finish_release_c1 = (stage_curr == STAGE_RELEASE_C1) & cube1_stacked
            if finish_release_c1.any():
                capture_retreat_target(finish_release_c1, retreat_target_1_b)
                stage[finish_release_c1] = STAGE_RETREAT_1
                stage_steps[finish_release_c1] = 20
                gripper_steps_left[finish_release_c1] = 20
                log_stage(finish_release_c1, STAGE_RETREAT_1, "cube1 stacked")

            # cube2 抓取链路
            finish_retreat_1 = (stage_curr == STAGE_RETREAT_1) & reached_retreat
            if finish_retreat_1.any():
                stage[finish_retreat_1] = STAGE_PRE_GRASP_C2
                stage_steps[finish_retreat_1] = 0
                log_stage(finish_retreat_1, STAGE_PRE_GRASP_C2, "retreat1 done")

            enter_grasp_c2 = (stage_curr == STAGE_PRE_GRASP_C2) & reached 
            if enter_grasp_c2.any():
                stage[enter_grasp_c2] = STAGE_GRASP_C2
                stage_steps[enter_grasp_c2] = 0
                gripper_steps_left[enter_grasp_c2] = args_cli.grasp_steps
                log_stage(enter_grasp_c2, STAGE_GRASP_C2, "reach cube2")

            finish_grasp_c2 = (stage_curr == STAGE_GRASP_C2) & (gripper_steps_left <= 0)
            if finish_grasp_c2.any():
                stage[finish_grasp_c2] = STAGE_LIFT_C2
                stage_steps[finish_grasp_c2] = 0
                log_stage(finish_grasp_c2, STAGE_LIFT_C2, "close gripper cube2")

            finish_lift_c2 = (stage_curr == STAGE_LIFT_C2) & cube2_lifted
            if finish_lift_c2.any():
                stage[finish_lift_c2] = STAGE_PRE_PLACE_C2
                stage_steps[finish_lift_c2] = 0
                log_stage(finish_lift_c2, STAGE_PRE_PLACE_C2, "cube2 lifted")

            enter_release_c2 = (stage_curr == STAGE_PRE_PLACE_C2) & place2_aligned & ee_is_slow
            if enter_release_c2.any():
                stage[enter_release_c2] = STAGE_RELEASE_C2
                stage_steps[enter_release_c2] = 0
                gripper_steps_left[enter_release_c2] = args_cli.release_steps
                log_stage(enter_release_c2, STAGE_RELEASE_C2, "arrive place2")

            finish_release_c2 = (stage_curr == STAGE_RELEASE_C2) & cube2_stacked
            if finish_release_c2.any():
                capture_retreat_target(finish_release_c2, retreat_target_2_b)
                stage[finish_release_c2] = STAGE_RETREAT_2
                stage_steps[finish_release_c2] = 0
                gripper_steps_left[finish_release_c2] = 0
                log_stage(finish_release_c2, STAGE_RETREAT_2, "cube2 stacked")

            finish_retreat_2 = (stage_curr == STAGE_RETREAT_2) & reached_retreat
            if finish_retreat_2.any():
                stage[finish_retreat_2] = STAGE_FINAL_HOLD
                stage_steps[finish_retreat_2] = 0
                hold_steps_left[finish_retreat_2] = final_hold_steps
                log_stage(finish_retreat_2, STAGE_FINAL_HOLD, "retreat2 done")

            finish_episode = (stage == STAGE_FINAL_HOLD) & (hold_steps_left <= 0)
            if finish_episode.any():
                reset_envs(finish_episode, "episode complete")
                obs = env.get_observations()
                continue

            # 阶段切换后，重新写一次 target_pose，保证当前步就跟踪新的 BT 目标。
            apply_bt_targets()

            # ---------- 4) 刷新观测，调用统一 reach policy ----------
            obs = env.get_observations()
            actions = policy(obs)

            # ---------- 5) BT 直接控制夹爪 ----------
            set_gripper_targets()

            # ---------- 6) 环境步进 ----------
            _, _, dones, _ = env.step(actions)

            done_mask = dones.to(torch.bool)
            if done_mask.any():
                policy_nn.reset(done_mask.to(dtype=torch.long))
                reset_bt_state(done_mask)
                log_stage(done_mask, STAGE_PRE_GRASP_C1, "env done")

            # ---------- 7) 更新计时器 ----------
            stage_steps += 1
            hold_steps_left[stage == STAGE_FINAL_HOLD] -= 1
            gripper_steps_left[(stage == STAGE_GRASP_C1) | (stage == STAGE_RELEASE_C1)] -= 1
            gripper_steps_left[(stage == STAGE_GRASP_C2) | (stage == STAGE_RELEASE_C2)] -= 1

        timestep += 1
        if args_cli.video and timestep == args_cli.video_length:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
