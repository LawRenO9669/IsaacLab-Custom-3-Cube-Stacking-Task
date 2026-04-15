import math

import isaaclab.utils.math as math_utils
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.managers import SceneEntityCfg


def set_target_pose_to_object(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str = "target_pose",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    height_offset: float = 0.05,
):
    obj = env.scene[asset_cfg.name]
    obj_pos_w = obj.data.root_pos_w[env_ids] - env.scene.env_origins[env_ids]
    obj_quat_w = obj.data.root_quat_w[env_ids]

    robot = env.scene["robot"]
    base_pos_w = robot.data.root_pos_w[env_ids] - env.scene.env_origins[env_ids]
    base_quat_w = robot.data.root_quat_w[env_ids]

    base_quat_inv = math_utils.quat_inv(base_quat_w)
    rel_pos_b = math_utils.quat_apply(base_quat_inv, obj_pos_w - base_pos_w)
    rel_quat_b = math_utils.quat_unique(math_utils.quat_mul(base_quat_inv, obj_quat_w))

    roll = torch.full((len(env_ids),), math.pi, device=rel_quat_b.device)
    roll_quat = math_utils.quat_from_euler_xyz(roll, torch.zeros_like(roll), torch.zeros_like(roll))
    rel_quat_b = math_utils.quat_mul(roll_quat, rel_quat_b)
    rel_pos_b[:, 2] += height_offset

    cmd = env.command_manager.get_term(command_name)
    cmd.pose_command_b[env_ids, :3] = rel_pos_b
    cmd.pose_command_b[env_ids, 3:] = rel_quat_b


class FixedPoseCommand(UniformPoseCommand):
    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)

        extras = {}
        for metric_name, metric_value in self.metrics.items():
            extras[metric_name] = torch.mean(metric_value[env_ids]).item()
            metric_value[env_ids] = 0.0

        self.command_counter[env_ids] = 0
        self.time_left[env_ids] = 1e9
        return extras
