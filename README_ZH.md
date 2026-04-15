# IsaacLab 自定义三方块堆叠任务

基于 `IsaacLab` 做的自定义任务整理，主要保留任务结构、`mdp` 设计、agent 配置和改过的 `play` 脚本。

<p align="center">
  <img src="assets/v2.gif" alt="V2 demo" width="600" />
</p>

这不是完整可复现项目，更像是一个 task-side reference repo。

## 内容

- `stack_cube`
  单方块抓取放置
- `stack_3cube_v1`
  双策略堆叠，`play_v1.py` 负责 checkpoint 切换
- `stack_3cube_v2`
  单个 `reach policy`，`play_v2.py` 负责阶段切换和夹爪控制

## 主要修改

- 在 IsaacLab 里注册自定义任务
- 组织 `env cfg / agent cfg / mdp`
- 修改官方 `play`
- reward 按任务阶段拆分

## Play 改动

### `play_v1.py`

- 加载两个 checkpoint
- 根据脚本判定在两个策略之间切换
- 插了撤手和异常重置逻辑

### `play_v2.py`

- 只保留一个低层 `reach policy`
- 高层用脚本改 `target_pose`
- 抓取、抬升、放置、松爪、撤手都在脚本里串

## Train

训练一般还是走 IsaacLab 官方入口，本仓库主要提供 task 和 config。

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Lift-Place-MyFanuc-v0
```

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task My-Stack-C3-v0
```

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task My-Reach-Stack-v0
```

## Play

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_v1.py \
  --task My-Stack-C3-Play-v0 \
  --checkpoint path/to/stage1_checkpoint.pt \
  --checkpoint_c2 path/to/stage2_checkpoint.pt \
  --num_envs 1
```

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_v2.py \
  --task My-Reach-Stack-Play-v0 \
  --checkpoint logs/rsl_rl/reach_stack/model_xxx.pt \
  --num_envs 1
```

## Reward

reward 不保证通用，只适合参考：

- task 拆分方式
- 稠密项 / 稀疏项怎么配
- 抓取、抬升、对位、释放、静止判定怎么拆 term

## 默认前提

默认已经熟悉 IsaacLab 这些东西：

- `gym.register`
- `ManagerBasedRLEnv`
- `SceneCfg / CommandsCfg / ActionsCfg / RewardsCfg`
- `SceneEntityCfg`
- IsaacLab 官方 `train.py / play.py`

如果这些概念还不熟，这仓库可能不太适合直接入门。

## 关于 USD

机器人 `USD` 因机密和版权原因不公开。

因此：

- 不保证开箱即用
- 不保证完整复现
- 更适合作为代码结构和任务实现思路参考

本项目仅供参考，默认读者已具备 IsaacLab 相关前置知识。

项目更新中...
