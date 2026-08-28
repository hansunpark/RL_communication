from __future__ import annotations

import csv
import json
import os

from collections import defaultdict

import numpy as np
import torch
from torch.distributions import Categorical

from gymnasium.spaces import (
    flatten,
    flatdim,
)

from env import (
    CooperativeTransportEnv,
)

from env.constants import (
    NO_MESSAGE,
)

from train.ppo import PPO

from .trajectory import (
    TrajectoryRecorder,
)


def flatten_obs(
    space,
    observation,
):
    return np.asarray(
        flatten(
            space,
            observation,
        ),
        dtype=np.float32,
    )


class PolicyEvaluator:

    def __init__(
        self,
        checkpoint,
        device=None,
    ):
        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else
                "cpu"
            )

        self.device = device

        self.env = (
            CooperativeTransportEnv(
                num_agents=4,
                vision_size=5,
                max_steps=200,
                curriculum_stage=6,
                reward_mode=(
                    "obstacle_shaping"
                ),
            )
        )

        sample_agent = (
            self.env.possible_agents[
                0
            ]
        )

        obs_dim = flatdim(
            self.env.observation_space(
                sample_agent
            )
        )

        self.ppo = PPO(
            obs_dim=obs_dim,
            num_actions=5,
            device=device,
        )

        self.ppo.load(
            checkpoint,
            load_optimizer=False,
        )

        self.ppo.network.eval()

    @torch.no_grad()
    def policy_action(
        self,
        agent,
        observation,
        deterministic=False,
    ):
        flat = flatten_obs(
            self.env.observation_space(agent),
            observation,
        )

        tensor = torch.tensor(
            flat,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        logits, _ = self.ppo.network(tensor)

        if deterministic:
            action = torch.argmax(
                logits,
                dim=-1,
            )
        else:
            distribution = Categorical(
                logits=logits
            )

            action = distribution.sample()

        return int(action.item())

    def run_episode(
        self,
        scenario,
        seed,
        record_trajectory=False,
        trajectory_path=None,
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        scenario_dict = (
            scenario.to_env_dict()
        )

        observations, _ = (
            self.env.reset(
                seed=seed,
                options={
                    "scenario":
                        scenario_dict
                },
            )
        )

        recorder = None

        if record_trajectory:

            recorder = (
                TrajectoryRecorder(
                    scenario_name=
                        scenario.name,

                    seed=seed,

                    width=
                        self.env.width,

                    height=
                        self.env.height,
                )
            )

            recorder.record_initial(
                self.env
            )

        episode_return = 0.0
        episode_length = 0

        target_moves = 0
        obstacle_moves = 0
        successful_pushes = 0

        success = False

        final_target_distance = None
        final_blocking = None

        while self.env.agents:

            current_agents = (
                self.env.agents[:]
            )

            actions = {}

            for agent in (
                current_agents
            ):
                physical_action = self.policy_action(
                    agent,
                    observations[agent],
                    deterministic=False,
                )

                actions[
                    agent
                ] = np.array(
                    [
                        physical_action,
                        NO_MESSAGE,
                    ],
                    dtype=np.int64,
                )

            (
                observations,
                rewards,
                terminations,
                truncations,
                infos,
            ) = self.env.step(
                actions
            )

            first_agent = (
                current_agents[0]
            )

            reward = (
                rewards[
                    first_agent
                ]
            )

            info = (
                infos[
                    first_agent
                ]
            )

            episode_return += reward
            episode_length += 1

            target_moves += int(
                info[
                    "target_moved"
                ]
            )

            obstacle_moves += int(
                info[
                    "obstacle_moves"
                ]
            )

            successful_pushes += int(
                info[
                    "successful_pushes"
                ]
            )

            success = bool(
                info["success"]
            )

            final_target_distance = (
                info[
                    "target_distance"
                ]
            )

            final_blocking = (
                info[
                    "obstacle_blocking"
                ]
            )

            if recorder is not None:

                recorder.record_step(
                    env=self.env,
                    step=episode_length,
                    actions=actions,
                    reward=reward,
                    info=info,
                )

        if recorder is not None:

            recorder.finish(
                success=success,
                episode_length=
                    episode_length,
                episode_return=
                    episode_return,
            )

            recorder.save(
                trajectory_path
            )

        return {
            "category":
                scenario.category,

            "scenario":
                scenario.name,

            "seed":
                int(seed),

            "success":
                int(success),

            "episode_length":
                episode_length,

            "return":
                episode_return,

            "target_moves":
                target_moves,

            "obstacle_moves":
                obstacle_moves,

            "successful_pushes":
                successful_pushes,

            "final_target_distance":
                final_target_distance,

            "final_obstacle_blocking":
                final_blocking,
        }

    def evaluate_scenario(
        self,
        scenario,
        episodes,
        seed_start,
        record_examples=False,
        trajectory_dir=None,
    ):
        results = []

        saved_success = 0
        saved_failure = 0

        for episode_idx in range(
            episodes
        ):
            seed = (
                seed_start
                +
                episode_idx
            )

            result = self.run_episode(
                scenario=scenario,
                seed=seed,
            )

            results.append(
                result
            )

            # -----------------------------------------
            # Save a few example trajectories
            # -----------------------------------------

            if record_examples:

                should_save = False

                if (
                    result["success"]
                    and
                    saved_success < 3
                ):
                    should_save = True
                    saved_success += 1

                elif (
                    not result["success"]
                    and
                    saved_failure < 3
                ):
                    should_save = True
                    saved_failure += 1

                if should_save:

                    outcome = (
                        "success"
                        if result["success"]
                        else
                        "failure"
                    )

                    filename = (
                        f"{scenario.name}_"
                        f"{outcome}_"
                        f"seed{seed}.json"
                    )

                    path = os.path.join(
                        trajectory_dir,
                        filename,
                    )

                    # Re-run deterministic episode.
                    self.run_episode(
                        scenario=scenario,
                        seed=seed,
                        record_trajectory=True,
                        trajectory_path=path,
                    )

        return results

    @staticmethod
    def summarize(
        results,
    ):
        if not results:
            return {}

        successes = [
            row
            for row in results
            if row["success"]
        ]

        success_rate = (
            len(successes)
            /
            len(results)
        )

        summary = {
            "episodes":
                len(results),

            "success_rate":
                success_rate,

            "mean_episode_length":
                float(
                    np.mean(
                        [
                            r[
                                "episode_length"
                            ]
                            for r
                            in results
                        ]
                    )
                ),

            "mean_return":
                float(
                    np.mean(
                        [
                            r["return"]
                            for r
                            in results
                        ]
                    )
                ),

            "mean_target_moves":
                float(
                    np.mean(
                        [
                            r[
                                "target_moves"
                            ]
                            for r
                            in results
                        ]
                    )
                ),

            "mean_obstacle_moves":
                float(
                    np.mean(
                        [
                            r[
                                "obstacle_moves"
                            ]
                            for r
                            in results
                        ]
                    )
                ),

            "mean_successful_pushes":
                float(
                    np.mean(
                        [
                            r[
                                "successful_pushes"
                            ]
                            for r
                            in results
                        ]
                    )
                ),
        }

        if successes:

            summary[
                "mean_success_length"
            ] = float(
                np.mean(
                    [
                        r[
                            "episode_length"
                        ]
                        for r
                        in successes
                    ]
                )
            )

        else:

            summary[
                "mean_success_length"
            ] = None

        return summary


def save_csv(
    rows,
    path,
):
    if not rows:
        return

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def save_json(
    data,
    path,
):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )