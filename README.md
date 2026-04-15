# IsaacLab Custom 3-Cube Stacking Task
Custom tasks built on top of `IsaacLab`, mainly kept for task structure, `mdp` design, agent config, and modified `play` scripts.

<p align="center">
  <img src="assets/v2.gif" alt="V2 demo" width="600" />
</p>

This is not a fully reproducible project. It is closer to a task-side reference repo.

## Contents

- `stack_cube`
  single-block pick-and-place
- `stack_3cube_v1`
  two-policy stacking, with checkpoint switching in `play_v1.py`
- `stack_3cube_v2`
  single `reach policy`, with stage switching and gripper control in `play_v2.py`

## Main Changes

- register custom tasks in IsaacLab
- organize `env cfg / agent cfg / mdp`
- modify the official `play`
- split reward terms by task stage

## Play Changes

### `play_v1.py`

- loads two checkpoints
- switches between two policies based on scripted checks
- adds retreat and reset logic

### `play_v2.py`

- keeps only one low-level `reach policy`
- rewrites `target_pose` in script
- scripts grasp, lift, place, release, and retreat

## Train

Training usually still goes through the official IsaacLab entry point. This repo mainly provides tasks and configs.

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

Reward design here is not meant to be general. It is mainly useful as reference for:

- task decomposition
- dense / sparse reward combinations
- grasp, lift, align, release, and static checks as separate terms

## Assumed Background

This repo assumes you are already familiar with:

- `gym.register`
- `ManagerBasedRLEnv`
- `SceneCfg / CommandsCfg / ActionsCfg / RewardsCfg`
- `SceneEntityCfg`
- official IsaacLab `train.py / play.py`

If not, this is probably not the right entry point.

## About The USD

The robot `USD` is not public due to confidentiality and copyright constraints.

Therefore:

- no out-of-the-box guarantee
- no full reproduction guarantee
- better treated as a reference for code structure and task implementation

This project is for reference only and assumes prior IsaacLab knowledge.

Work in progress...
