# RL_communication

[한국어](README.ko.md)

Multi-agent reinforcement learning sandbox: N agents cooperatively push
rectangular objects across a grid onto a goal area, trained with PPO through
a hand-designed curriculum (stage 0 → 7) plus out-of-distribution benchmarking.

## Overview

- **Environment**: `CooperativeTransportEnv` (`env/cooperative_transport_env.py`)
  is a [PettingZoo](https://pettingzoo.farley.ml/) `ParallelEnv` on a fixed-size
  grid. Agents must push a `target` object onto a set of `goal_cells`, possibly
  navigating around `obstacle` objects. Pushing only succeeds when all agents
  required to contact a face of the object move in the same direction on the
  same step (see `_resolve_movement`).
- **Observations** are per-agent and partially observable: a local
  `vision_size × vision_size` window (`local_map`, `object_ids`, `goal_mask`),
  the agent's global `position`, per-object `touching_objects` info, and a
  `messages` channel carrying relative positions broadcast by other agents.
- **Actions** are `MultiDiscrete([5, 2])` per agent: a movement action
  (`STAY/UP/RIGHT/DOWN/LEFT`) and a communication action (`NO_MESSAGE`/`HELP`).
  When an agent emits `HELP`, every other agent receives its relative offset
  in the next observation's `messages` field. The current curriculum training
  scripts always send `NO_MESSAGE` — the communication channel is implemented
  and observable, but not yet trained on.
- **Training**: `train/ppo.py` is a standard clipped-objective PPO with a
  shared MLP `ActorCritic` (`train/network.py`) and a generalized-advantage
  rollout buffer (`train/rollout_buffer.py`). All agents share one policy.
- **Curriculum**: training runs through progressively harder scenarios
  (`train/stages/stage_0_6`, then `train/stages/stage_7`), automatically
  advancing once a rolling success-rate threshold is met.
- **Evaluation**: `evaluation/` runs a frozen checkpoint against in-distribution
  and out-of-distribution benchmark scenarios (shifted/obstacle/goal
  perturbations, combined and stress tests), producing CSV/JSON summaries and
  optional episode trajectories for later visualization.

## Project layout

```
env/                      CooperativeTransportEnv, RectObject, constants
train/
  network.py               ActorCritic (shared policy/value MLP)
  ppo.py                    PPO update loop, act/value, save/load
  rollout_buffer.py         Multi-agent GAE rollout buffer
  stages/
    common.py               set_seed, flatten_obs helpers shared by all stages
    stage_0_6/               Stage 0-6 curriculum manager + training script
    stage_7/                 Stage 7 (lane-shift generalization) curriculum,
                             training script, left-lane fine-tuning, and a
                             per-lane diagnostic script
  evaluate.py               Ad-hoc single-checkpoint evaluation helper
evaluation/
  benchmark_scenarios.py    IID / obstacle / target-goal / combined / stress
                             scenario definitions
  evaluator.py              PolicyEvaluator: runs episodes, aggregates metrics
  run_baseline_benchmark.py CLI entry point for full benchmark sweeps
  trajectory.py             Episode trajectory recording
  visualize.py               Rendering trajectories (e.g. to GIF)
checkpoints/               Saved PPO checkpoints per stage/milestone
evaluation_results/         CSV/JSON benchmark outputs, recorded trajectories
runs/                      TensorBoard event logs
logs/                      Text training logs
tests/test_env.py          PettingZoo API compliance + environment unit tests
main.py                    Minimal random-policy smoke test / render demo
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.12-compatible `numpy`, `gymnasium`, `pettingzoo`, `torch`,
`tensorboard`, `matplotlib`, `pillow`, and `pytest` for tests.

## Running things

**Smoke-test the environment with random actions (renders to console):**

```bash
python main.py
```

**Run the environment unit tests:**

```bash
pytest tests/test_env.py
```

**Train Stage 0-6 curriculum from scratch:**

```bash
python -m train.stages.stage_0_6.train
```

Writes TensorBoard logs to `runs/curriculum_v2`, periodic checkpoints to
`checkpoints/episode_*.pt`, and per-stage mastery checkpoints
(`stage_N_mastered.pt`).

**Fine-tune Stage 7 (lane-shift generalization) from a mastered Stage 6 checkpoint:**

```bash
python -m train.stages.stage_7.train
```

Loads `checkpoints/stage_6_mastered.pt` and progressively shifts the
target/goal/obstacle lane position (`stage_7A_shift1` → `stage_7B_shift2` →
`stage_7C_all_lanes`), tracking success rate independently per substage.

**Diagnose per-lane weaknesses / fine-tune on weak lanes:**

```bash
python -m train.stages.stage_7.lane_diagnostic
python -m train.stages.stage_7.finetune_left_lanes
```

**Benchmark a checkpoint against IID + OOD scenario suites:**

```bash
python -m evaluation.run_baseline_benchmark --checkpoint checkpoints/stage_7_mastered.pt --mode deterministic
```

Results land in `evaluation_results/<mode>/` (`baseline_episodes.csv`,
`baseline_summary.json`, and recorded trajectories for representative
successes/failures).

## Curriculum stages (0-6)

Each stage widens the spawn distribution and/or introduces obstacles; the
`CurriculumManager` (`train/stages/stage_0_6/curriculum.py`) advances a stage
once a rolling 100-episode success rate exceeds 0.85 (with a minimum episode
count per stage):

| Stage | Change |
|---|---|
| 0 | Fixed agent/target/goal layout, no obstacles |
| 1-2 | Agents spawn randomly near the target; larger target/goal area |
| 3 | Spawn radius widens in sub-levels (radius 3 → 5 → fully random) |
| 4 | First obstacle introduced; reward shaping toward pushing it out of the corridor kicks in only if target-only reward fails to reach mastery |
| 5 | Fully random agent spawn with the obstacle present |
| 6 | Second obstacle added (final stage of the base curriculum) |

Stage 7 (`train/stages/stage_7/curriculum.py`) then tests generalization by
shifting the entire Stage 6 geometry sideways across the map's lanes.

## Notes

- Object pushes require every agent adjacent to the pushed face to choose the
  same direction in the same step; conflicting pushes/agent moves are
  resolved via `_resolve_agent_conflicts` / `_remove_object_conflicts` in
  `env/cooperative_transport_env.py`.
- Reward = step penalty + distance-to-goal shaping (+ optional obstacle
  corridor-clearing shaping in `obstacle_shaping` mode) + a terminal success
  bonus, shared identically across all agents.
- `env.state()` returns a flattened global grid (useful for a centralized
  critic or debugging), though the current PPO setup only uses per-agent
  observations.
