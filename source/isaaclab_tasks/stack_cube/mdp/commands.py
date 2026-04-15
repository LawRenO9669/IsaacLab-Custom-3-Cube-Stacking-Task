import math
import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand

def set_target_pose_to_object(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str = "target_pose",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """在 reset 时把 target_pose 设为当前物体位姿（转换到机器人 base 坐标系）."""
    # 取物体世界系位姿
    obj = env.scene[asset_cfg.name]
    obj_pos_w = obj.data.root_pos_w[env_ids] - env.scene.env_origins[env_ids]
    obj_quat_w = obj.data.root_quat_w[env_ids]

    # 取机器人 base 世界系位姿
    robot = env.scene["robot"]
    base_pos_w = robot.data.root_pos_w[env_ids] - env.scene.env_origins[env_ids]
    base_quat_w = robot.data.root_quat_w[env_ids]

    # 世界 -> base 坐标变换
    base_quat_inv = math_utils.quat_inv(base_quat_w)
    # 位置先做平移再旋转
    rel_pos_b = math_utils.quat_apply(base_quat_inv, obj_pos_w - base_pos_w)
    # 姿态做相对旋转
    rel_quat_b = math_utils.quat_mul(base_quat_inv, obj_quat_w)
    rel_quat_b = math_utils.quat_unique(rel_quat_b)
    # 强制 roll = pi，确保姿态有 180° 翻转
    roll = torch.full((len(env_ids),), math.pi, device=rel_quat_b.device)
    zero = torch.zeros_like(roll)
    roll_quat = math_utils.quat_from_euler_xyz(roll, zero, zero)
    rel_quat_b = math_utils.quat_mul(roll_quat, rel_quat_b)
    # 将目标放在物体上方 5cm
    rel_pos_b[:, 2] += 0.05
    
    # 写入 command buffer（UniformPoseCommand.pose_command_b）
    cmd = env.command_manager.get_term(command_name)
    cmd.pose_command_b[env_ids, :3] = rel_pos_b
    cmd.pose_command_b[env_ids, 3:] = rel_quat_b
    # print("event set_target_pose:", cmd.pose_command_b[env_ids[0]])


class FixedPoseCommand(UniformPoseCommand):
    """避免 reset 时自动重采样的 Pose 命令。

    `UniformPoseCommand.reset()` 会重采样目标姿态，这会覆盖 reset 事件中写入的值。
    这里覆写 reset：仅重置 metric 和计数，不调用 `_resample_command`。
    """

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)

        extras = {}
        for metric_name, metric_value in self.metrics.items():
            extras[metric_name] = torch.mean(metric_value[env_ids]).item()
            metric_value[env_ids] = 0.0

        self.command_counter[env_ids] = 0
        # 设置极大时间避免自动重采样；目标姿态由事件手动写入。
        self.time_left[env_ids] = 1e9
        return extras
