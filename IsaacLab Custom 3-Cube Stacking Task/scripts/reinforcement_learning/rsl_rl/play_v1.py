# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause




import argparse
import os
import sys
import time

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play stack_3cube_v1 with behavior-tree switching.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

parser.add_argument("--checkpoint_c2", type=str, required=True, help="Checkpoint path for stage-2 (red cube) policy.")

# Behavior-tree tuning
parser.add_argument("--reload_interval_s", type=float, default=3.0, help="Reload active stage policy every N seconds until success.")
parser.add_argument("--retreat_steps", type=int, default=50, help="Retreat steps after C1 success.")
parser.add_argument("--final_hold_s", type=float, default=3.0, help="Hold duration in seconds after C2 success.")
parser.add_argument("--drop_z", type=float, default=0.0, help="Custom drop threshold: z < drop_z triggers reset to C1.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

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
from isaaclab_tasks.stack_3cube_v1 import mdp
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401

STAGE_C1 = 0
STAGE_C1_RETREAT = 1
STAGE_C2 = 2
STAGE_FINAL_HOLD = 3
STAGE_NAMES = {
    STAGE_C1: "c1",
    STAGE_C1_RETREAT: "return1",
    STAGE_C2: "c2",
    STAGE_FINAL_HOLD: "return2",
}

RETREAT_POSE = [0.60, -1.10, 0.56, 0.00, -0.08, -0.09]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_c1 = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_c1 = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    if resume_c1 is None:
        raise FileNotFoundError("No stage-1 checkpoint found. Pass --checkpoint or configure load_run/load_checkpoint.")

    resume_c2 = retrieve_file_path(args_cli.checkpoint_c2)
    if resume_c2 is None:
        raise FileNotFoundError(f"Invalid --checkpoint_c2 path: {args_cli.checkpoint_c2}")

    env_cfg.log_dir = os.path.dirname(resume_c1)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(env_cfg.log_dir, "videos", "play_BT"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    def build_runner(path: str):
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic
        return runner, policy, policy_nn

    def load_policy(path: str, stage_name: str):
        print(f"[INFO] Loading {stage_name} checkpoint: {path}")
        return build_runner(path)

    runner_c1, policy_c1, policy_nn_c1 = load_policy(resume_c1, "C1")
    runner_c2, policy_c2, policy_nn_c2 = load_policy(resume_c2, "C2")

    action_manager = env.unwrapped.action_manager
    robot = env.unwrapped.scene["robot"]
    arm_term = action_manager.get_term("arm_action")
    arm_joint_ids, _ = robot.find_joints(arm_term._joint_names, preserve_order=True)
    q_retreat = torch.tensor(RETREAT_POSE, device=env.device, dtype=robot.data.joint_pos.dtype).unsqueeze(0)

    action_slices = {}
    idx = 0
    for name, dim in zip(action_manager.active_terms, action_manager.action_term_dim):
        action_slices[name] = slice(idx, idx + dim)
        idx += dim
    arm_action_slice = action_slices["arm_action"]
    gripper_action_slice = action_slices["gripper_action"]
    arm_col_1 = arm_action_slice.start + 0
    arm_col_2 = arm_action_slice.start + 1

    stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    stage_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    retreat_steps_left = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    hold_steps_left = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    dt = env.unwrapped.step_dt
    reload_steps = max(1, int(args_cli.reload_interval_s / dt))
    hold_steps = max(1, int(args_cli.final_hold_s / dt))

    def reset_bt_state(mask: torch.Tensor):
        stage[mask] = STAGE_C1
        stage_steps[mask] = 0
        retreat_steps_left[mask] = 0
        hold_steps_left[mask] = 0

    def log_stage(mask: torch.Tensor, stage_id: int, reason: str):
        ids = torch.where(mask)[0]
        if ids.numel() == 0:
            return
        ids_preview = ids[:8].tolist()
        suffix = "..." if ids.numel() > 8 else ""
        print(f"[STAGE] {reason}: {STAGE_NAMES[stage_id]} envs={ids_preview}{suffix}")

    def reset_envs(mask: torch.Tensor):
        ids = torch.where(mask)[0]
        if ids.numel() == 0:
            return
        env.unwrapped.reset(env_ids=ids)
        reset_bt_state(mask)
        done_vec = mask.to(dtype=torch.long)
        policy_nn_c1.reset(done_vec)
        policy_nn_c2.reset(done_vec)
        log_stage(mask, STAGE_C1, "reset")

    obs = env.get_observations()
    timestep = 0
    log_stage(torch.ones(env.num_envs, dtype=torch.bool, device=env.device), STAGE_C1, "init")

    # 主循环：每个仿真步都执行“观测 -> 决策树更新 -> 动作生成 -> 步进 -> 计时”
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            # ---------- 1) 计算决策信号 ----------
            # c1_success / c2_success:
            # 分别表示 blue(c1)、red(c2) 是否满足“放置成功”判据。
            # 这些信号决定状态机从 C1->撤手->C2->最终保持 的流转。
            c1_success = mdp.success_at_goal_xy_static(
                env.unwrapped,
                command_name="target_pose",
                threshold=0.03,
                velocity_threshold=0.05,
                open_threshold=0.032,
                height_min=0.01,
                gripper_joint_names="panda_finger_joint[1-2]",
                object_cfg=SceneEntityCfg("cube1"),
            )
            c2_success = mdp.success_at_goal_xy_static(
                env.unwrapped,
                command_name="target_pose_c2",
                threshold=0.03,
                velocity_threshold=0.05,
                open_threshold=0.032,
                height_min=0.10,
                gripper_joint_names="panda_finger_joint[1-2]",
                object_cfg=SceneEntityCfg("cube2"),
            )

            # drop_any:
            # 任意一个方块低于高度阈值都算“掉落”。
            # 一旦触发，会在本步将该 env 强制 reset 回 C1 阶段。
            cube1_z = env.unwrapped.scene["cube1"].data.root_pos_w[:, 2]
            cube2_z = env.unwrapped.scene["cube2"].data.root_pos_w[:, 2]
            drop_any = (cube1_z < args_cli.drop_z) | (cube2_z < args_cli.drop_z)

            # 每 3 秒重载一次当前阶段权重，但不重置环境。
            # 这样在单回合任务时间内，策略会持续尝试直到堆叠成功。
            reload_c1 = (stage == STAGE_C1) & (~c1_success) & (stage_steps > 0) & ((stage_steps % reload_steps) == 0)
            reload_c2 = (stage == STAGE_C2) & (~c2_success) & (stage_steps > 0) & ((stage_steps % reload_steps) == 0)
            if reload_c1.any():
                runner_c1, policy_c1, policy_nn_c1 = load_policy(resume_c1, "C1")
                policy_nn_c1.reset(reload_c1.to(dtype=torch.long))
                print(f"[INFO] Reloaded C1 checkpoint for {int(reload_c1.sum().item())} envs.")
            if reload_c2.any():
                runner_c2, policy_c2, policy_nn_c2 = load_policy(resume_c2, "C2")
                policy_nn_c2.reset(reload_c2.to(dtype=torch.long))
                print(f"[INFO] Reloaded C2 checkpoint for {int(reload_c2.sum().item())} envs.")

            # force_reset:
            # 这里只保留真正需要立即重置的掉落情况。
            # 阶段超时不再 reset，而是通过上面的定时重载继续尝试。
            force_reset = drop_any
            if force_reset.any():
                reset_envs(force_reset)
                # 局部 reset 后主动刷新观测，保证后续策略前向拿到的是最新状态。
                obs = env.get_observations()
                continue

            # ---------- 2) 状态机转移 ----------
            # C1 成功 -> C1 撤手阶段
            enter_c1_retreat = (stage == STAGE_C1) & c1_success
            stage[enter_c1_retreat] = STAGE_C1_RETREAT
            stage_steps[enter_c1_retreat] = 0
            retreat_steps_left[enter_c1_retreat] = args_cli.retreat_steps
            log_stage(enter_c1_retreat, STAGE_C1_RETREAT, "c1 success")

            # C1 撤手完成 -> C2 阶段
            finish_c1_retreat = (stage == STAGE_C1_RETREAT) & (retreat_steps_left <= 0)
            stage[finish_c1_retreat] = STAGE_C2
            stage_steps[finish_c1_retreat] = 0
            log_stage(finish_c1_retreat, STAGE_C2, "return1 done")

            # C2 成功 -> return2 阶段（保持期间继续执行撤手姿态）
            enter_final = (stage == STAGE_C2) & c2_success
            stage[enter_final] = STAGE_FINAL_HOLD
            stage_steps[enter_final] = 0
            hold_steps_left[enter_final] = hold_steps
            log_stage(enter_final, STAGE_FINAL_HOLD, "c2 success")

            # return2 持续 3 秒后，重置该 env，重新开始 play。
            finish_final = (stage == STAGE_FINAL_HOLD) & (hold_steps_left <= 0)
            if finish_final.any():
                reset_envs(finish_final)
                obs = env.get_observations()

            # ---------- 3) 动作合成 ----------
            # 两个策略都前向一次，然后按 stage 选择每个 env 的动作来源：
            # - stage==C2: 使用 policy_c2
            # - 其他阶段: 默认用 policy_c1（随后脚本阶段会被覆盖）
            act_c1 = policy_c1(obs)
            act_c2 = policy_c2(obs)

            use_c2 = (stage == STAGE_C2).float().unsqueeze(-1)
            actions = act_c1 * (1.0 - use_c2) + act_c2 * use_c2

            # 脚本撤手覆盖（与原脚本风格一致）：
            # - C1 撤手阶段：前半程优先第2关节，后半程第1/第2关节共同收敛
            # - 最终保持阶段：维持后半程的第1/第2关节控制并保持夹爪张开
            retreat_ids = torch.where(stage == STAGE_C1_RETREAT)[0]
            if retreat_ids.numel() > 0:
                actions[retreat_ids, arm_action_slice] = 0.0
                first_half_mask = retreat_steps_left[retreat_ids] > (args_cli.retreat_steps // 2)
                if first_half_mask.any():
                    ids_a = retreat_ids[first_half_mask]
                    q_curr_a = robot.data.joint_pos[ids_a][:, arm_joint_ids]
                    err_j2 = q_retreat[0, 1] - q_curr_a[:, 1]
                    actions[ids_a, arm_col_2] = torch.clamp(err_j2 / 0.1, -10.0, 10.0)
                second_half_mask = ~first_half_mask
                if second_half_mask.any():
                    ids_b = retreat_ids[second_half_mask]
                    q_curr_b = robot.data.joint_pos[ids_b][:, arm_joint_ids]
                    err_j1 = q_retreat[0, 0] - q_curr_b[:, 0]
                    err_j2 = q_retreat[0, 1] - q_curr_b[:, 1]
                    actions[ids_b, arm_col_1] = torch.clamp(err_j1 / 0.1, -10.0, 1.0)
                    actions[ids_b, arm_col_2] = torch.clamp(err_j2 / 0.1, -10.0, 10.0)
                actions[retreat_ids, gripper_action_slice] = 1.0

            hold_ids = torch.where(stage == STAGE_FINAL_HOLD)[0]
            if hold_ids.numel() > 0:
                actions[hold_ids, arm_action_slice] = 0.0
                q_curr_h = robot.data.joint_pos[hold_ids][:, arm_joint_ids]
                err_j1_h = q_retreat[0, 0] - q_curr_h[:, 0]
                err_j2_h = q_retreat[0, 1] - q_curr_h[:, 1]
                actions[hold_ids, arm_col_1] = torch.clamp(err_j1_h / 0.1, -10.0, 1.0)
                actions[hold_ids, arm_col_2] = torch.clamp(err_j2_h / 0.1, -10.0, 10.0)
                actions[hold_ids, gripper_action_slice] = 1.0

            # ---------- 4) 环境步进 ----------
            # step 后由环境自身处理 terminations；dones 反映本步是否被环境重置。
            obs, _, dones, _ = env.step(actions)

            # 对 done 的 env 同步清理脚本状态，避免旧阶段残留到下一回合。
            done_mask = dones.to(torch.bool)
            if done_mask.any():
                done_vec = done_mask.to(dtype=torch.long)
                policy_nn_c1.reset(done_vec)
                policy_nn_c2.reset(done_vec)
                reset_bt_state(done_mask)
                log_stage(done_mask, STAGE_C1, "env done")

            # ---------- 5) 计时器更新 ----------
            # stage_steps: 仅在 C1/C2 任务阶段累加（用于失败超时判定）
            # retreat_steps_left / hold_steps_left: 在各自阶段递减至 0 触发转移
            stage_steps[(stage == STAGE_C1) | (stage == STAGE_C2)] += 1
            retreat_steps_left[stage == STAGE_C1_RETREAT] -= 1
            hold_steps_left[stage == STAGE_FINAL_HOLD] -= 1

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
