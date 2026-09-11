import csv
import json
import os

from collections import defaultdict

import numpy as np
import torch

from gymnasium.spaces.utils import (
    flatten,
    flatdim,
)

from torch.distributions import (
    Categorical,
)

from env.cooperative_transport_env import (
    CooperativeTransportEnv,
)

from env.constants import (
    NO_MESSAGE,
)

from train.ppo import PPO

from evaluation.trajectory import (
    TrajectoryRecorder,
)


class PolicyEvaluator:

    def __init__(
        self,
        checkpoint,
        device=None,
        action_mode="stochastic",
    ):

        # ====================================================
        # Device
        # ====================================================

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(
            device
        )

        # ====================================================
        # Action mode
        # ====================================================

        if action_mode not in [
            "stochastic",
            "deterministic",
        ]:
            raise ValueError(
                "action_mode must be "
                "'stochastic' or "
                "'deterministic'"
            )

        self.action_mode = (
            action_mode
        )

        # ====================================================
        # Environment
        #
        # 중요:
        # 네 환경은 __init__(stage=...)
        # 형태가 아니다.
        # ====================================================

        self.env = (
            CooperativeTransportEnv()
        )

        self.env.set_curriculum(
            stage=6,
            spawn_level=2,
            reward_mode=
                "obstacle_shaping",
        )

        # ====================================================
        # Observation dimension
        # ====================================================

        sample_agent = (
            self.env.possible_agents[0]
        )

        observation_space = (
            self.env.observation_space(
                sample_agent
            )
        )

        obs_dim = flatdim(
            observation_space
        )

        # ====================================================
        # PPO
        # ====================================================

        self.ppo = PPO(
            obs_dim=obs_dim,
            num_actions=5,
            device=self.device,
        )

        # PPO.load() 시그니처가
        # 현재 코드 버전에 따라 다를 수 있으므로
        # 두 방식 모두 대응
        try:
            self.ppo.load(
                checkpoint,
                load_optimizer=False,
            )

        except TypeError:
            self.ppo.load(
                checkpoint
            )

        self.ppo.network.eval()

        print(
            "=" * 60
        )

        print(
            f"Device      : "
            f"{self.device}"
        )

        print(
            f"Action mode : "
            f"{self.action_mode}"
        )

        print(
            f"Checkpoint  : "
            f"{checkpoint}"
        )

        print(
            f"Obs dim     : "
            f"{obs_dim}"
        )

        print(
            "=" * 60
        )

    # ========================================================
    # Observation → Tensor
    # ========================================================

    def _observation_to_tensor(
        self,
        agent,
        observation,
    ):

        flat_observation = flatten(
            self.env.observation_space(
                agent
            ),
            observation,
        )

        tensor = torch.tensor(
            flat_observation,
            dtype=torch.float32,
            device=self.device,
        )

        return tensor.unsqueeze(0)

    # ========================================================
    # Policy
    # ========================================================

    @torch.no_grad()
    def select_action(
        self,
        agent,
        observation,
    ):

        obs_tensor = (
            self._observation_to_tensor(
                agent,
                observation,
            )
        )

        logits, _ = (
            self.ppo.network(
                obs_tensor
            )
        )

        probabilities = (
            torch.softmax(
                logits,
                dim=-1,
            )
        )

        if (
            self.action_mode
            == "deterministic"
        ):

            action = torch.argmax(
                logits,
                dim=-1,
            )

        else:

            distribution = (
                Categorical(
                    logits=logits
                )
            )

            action = (
                distribution.sample()
            )

        action_int = int(
            action.item()
        )

        probabilities_np = (
            probabilities
            .squeeze(0)
            .cpu()
            .numpy()
        )

        return (
            action_int,
            probabilities_np,
        )

    # ========================================================
    # Environment reset
    # ========================================================

    def _reset_environment(
        self,
        scenario,
        seed,
    ):

        # ----------------------------------------------------
        # IID
        #
        # config=None이면 원래 Stage 6를 그대로 사용.
        # custom scenario를 만들지 않는다.
        # ----------------------------------------------------

        if scenario.config is None:

            self.env.set_curriculum(
                stage=6,
                spawn_level=2,
                reward_mode=
                    "obstacle_shaping",
            )

            observations, infos = (
                self.env.reset(
                    seed=seed
                )
            )

            return (
                observations,
                infos,
            )

        # ----------------------------------------------------
        # OOD custom scenario
        # ----------------------------------------------------

        observations, infos = (
            self.env.reset(
                seed=seed,
                options={
                    "scenario":
                        scenario.to_env_dict()
                },
            )
        )

        return (
            observations,
            infos,
        )

    # ========================================================
    # Single episode
    # ========================================================

    @torch.no_grad()
    def run_episode(
        self,
        scenario,
        seed,
        record_trajectory=False,
        trajectory_path=None,
    ):

        # ====================================================
        # Random seed
        #
        # 환경 randomness뿐 아니라
        # stochastic policy sampling도 재현 가능하게 함.
        # ====================================================

        np.random.seed(
            seed
        )

        torch.manual_seed(
            seed
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                seed
            )

        # ====================================================
        # Reset
        # ====================================================

        (
            observations,
            _,
        ) = self._reset_environment(
            scenario,
            seed,
        )

        episode_return = 0.0

        episode_length = 0

        episode_target_moves = 0

        episode_obstacle_moves = 0

        episode_successful_pushes = 0

        success = False

        last_info = {}

        # ====================================================
        # Trajectory recorder
        # ====================================================

        recorder = None

        if record_trajectory:

            recorder = (
                TrajectoryRecorder(
                    scenario_name=
                        scenario.name,

                    seed=
                        seed,

                    width=
                        self.env.width,

                    height=
                        self.env.height,

                    action_mode=
                        self.action_mode,
                )
            )

        # ====================================================
        # Episode loop
        # ====================================================

        while self.env.agents:

            current_agents = list(
                self.env.agents
            )

            actions = {}

            action_probabilities = {}

            # ------------------------------------------------
            # Agent actions
            # ------------------------------------------------

            for agent in current_agents:

                (
                    physical_action,
                    probabilities,
                ) = self.select_action(
                    agent,
                    observations[agent],
                )

                actions[agent] = (
                    np.array(
                        [
                            physical_action,
                            NO_MESSAGE,
                        ],
                        dtype=np.int64,
                    )
                )

                action_probabilities[
                    agent
                ] = (
                    probabilities.tolist()
                )

            # ------------------------------------------------
            # Environment step
            # ------------------------------------------------

            (
                next_observations,
                rewards,
                terminations,
                truncations,
                infos,
            ) = self.env.step(
                actions
            )

            # ------------------------------------------------
            # Team reward
            #
            # 모든 agent가 공유 reward를 받으므로
            # agent 수만큼 더하면 안 된다.
            # ------------------------------------------------

            step_reward = 0.0

            if current_agents:

                first_agent = (
                    current_agents[0]
                )

                step_reward = float(
                    rewards.get(
                        first_agent,
                        0.0,
                    )
                )

                last_info = (
                    infos.get(
                        first_agent,
                        {},
                    )
                )

                episode_return += (
                    step_reward
                )

                episode_target_moves += (
                    int(
                        last_info.get(
                            "target_moved",
                            0,
                        )
                    )
                )

                episode_obstacle_moves += (
                    int(
                        last_info.get(
                            "obstacle_moves",
                            0,
                        )
                    )
                )

                episode_successful_pushes += (
                    int(
                        last_info.get(
                            "successful_pushes",
                            0,
                        )
                    )
                )

                success = bool(
                    last_info.get(
                        "success",
                        False,
                    )
                )

            episode_length += 1

            # ------------------------------------------------
            # Trajectory
            # ------------------------------------------------

            if recorder is not None:

                recorder.record_frame(
                    env=self.env,
                    actions=actions,
                    reward=step_reward,
                    info=last_info,
                    action_probabilities=
                        action_probabilities,
                )

            observations = (
                next_observations
            )

            # 안전 장치
            if (
                episode_length
                >= self.env.max_steps
            ):
                break

        # ====================================================
        # Save trajectory
        # ====================================================

        if recorder is not None:

            recorder.finish(
                success=
                    success,

                episode_length=
                    episode_length,

                episode_return=
                    episode_return,
            )

            if trajectory_path:
                recorder.save(
                    trajectory_path
                )

        # ====================================================
        # Episode result
        # ====================================================

        return {
            "scenario":
                scenario.name,

            "seed":
                int(seed),

            "action_mode":
                self.action_mode,

            "success":
                int(success),

            "episode_length":
                int(
                    episode_length
                ),

            "return":
                float(
                    episode_return
                ),

            "target_moves":
                int(
                    episode_target_moves
                ),

            "obstacle_moves":
                int(
                    episode_obstacle_moves
                ),

            "successful_pushes":
                int(
                    episode_successful_pushes
                ),

            "final_target_distance":
                last_info.get(
                    "target_distance",
                    None,
                ),

            "final_obstacle_blocking":
                last_info.get(
                    "obstacle_blocking",
                    None,
                ),
        }

    # ========================================================
    # Multiple episodes
    # ========================================================

    def evaluate_scenarios(
        self,
        scenarios,
        episodes_per_scenario,
        seed_start=100000,
        trajectory_dir=None,
        max_success_trajectories=3,
        max_failure_trajectories=3,
    ):

        all_results = []

        seed_counter = (
            seed_start
        )

        for scenario in scenarios:

            print()
            print(
                "=" * 70
            )

            print(
                f"Scenario : "
                f"{scenario.name}"
            )

            print(
                f"Mode     : "
                f"{self.action_mode}"
            )

            print(
                "=" * 70
            )

            scenario_results = []

            saved_successes = 0

            saved_failures = 0

            for episode in range(
                episodes_per_scenario
            ):

                seed = seed_counter

                seed_counter += 1

                result = (
                    self.run_episode(
                        scenario=
                            scenario,

                        seed=
                            seed,

                        record_trajectory=
                            False,
                    )
                )

                all_results.append(
                    result
                )

                scenario_results.append(
                    result
                )

                # --------------------------------------------
                # Representative trajectories
                #
                # stochastic도 같은 seed를 다시 주므로
                # 동일 action sample sequence가 재현된다.
                # --------------------------------------------

                record_this = False

                label = None

                if (
                    result["success"]
                    and
                    saved_successes
                    <
                    max_success_trajectories
                ):
                    record_this = True

                    label = "success"

                    saved_successes += 1

                elif (
                    not result["success"]
                    and
                    saved_failures
                    <
                    max_failure_trajectories
                ):
                    record_this = True

                    label = "failure"

                    saved_failures += 1

                if (
                    record_this
                    and
                    trajectory_dir
                    is not None
                ):

                    os.makedirs(
                        trajectory_dir,
                        exist_ok=True,
                    )

                    trajectory_path = (
                        os.path.join(
                            trajectory_dir,

                            f"{scenario.name}_"
                            f"{label}_"
                            f"seed{seed}.json"
                        )
                    )

                    self.run_episode(
                        scenario=
                            scenario,

                        seed=
                            seed,

                        record_trajectory=
                            True,

                        trajectory_path=
                            trajectory_path,
                    )

                # --------------------------------------------
                # progress print
                # --------------------------------------------

                if (
                    (episode + 1)
                    % 20
                    == 0
                    or
                    episode + 1
                    ==
                    episodes_per_scenario
                ):

                    success_rate = (
                        np.mean(
                            [
                                r["success"]
                                for r
                                in scenario_results
                            ]
                        )
                    )

                    print(
                        f"Episode "
                        f"{episode + 1:4d} / "
                        f"{episodes_per_scenario} "
                        f"| success="
                        f"{success_rate:.3f}"
                    )

            summary = (
                self.aggregate(
                    scenario_results
                )
            )

            print()

            print(
                f"Success rate          : "
                f"{summary['success_rate']:.3f}"
            )

            print(
                f"Average length        : "
                f"{summary['mean_episode_length']:.2f}"
            )

            print(
                f"Average return        : "
                f"{summary['mean_return']:.3f}"
            )

            print(
                f"Average target moves  : "
                f"{summary['mean_target_moves']:.2f}"
            )

            print(
                f"Average obstacle moves: "
                f"{summary['mean_obstacle_moves']:.2f}"
            )

        return all_results

    # ========================================================
    # Aggregate
    # ========================================================

    @staticmethod
    def aggregate(
        results
    ):

        if not results:
            return {
                "episodes": 0,

                "success_rate":
                    0.0,

                "mean_episode_length":
                    None,

                "mean_return":
                    None,

                "mean_target_moves":
                    None,

                "mean_obstacle_moves":
                    None,

                "mean_successful_pushes":
                    None,

                "mean_success_length":
                    None,
            }

        successful = [
            r
            for r in results
            if r["success"] == 1
        ]

        if successful:
            mean_success_length = (
                float(
                    np.mean(
                        [
                            r[
                                "episode_length"
                            ]
                            for r
                            in successful
                        ]
                    )
                )
            )

        else:
            mean_success_length = (
                None
            )

        return {
            "episodes":
                len(results),

            "success_rate":
                float(
                    np.mean(
                        [
                            r["success"]
                            for r
                            in results
                        ]
                    )
                ),

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

            "mean_success_length":
                mean_success_length,
        }

    # ========================================================
    # Build summary
    # ========================================================

    def build_summary(
        self,
        grouped_results,
    ):

        output = {}

        for (
            group_name,
            results,
        ) in (
            grouped_results.items()
        ):

            scenario_groups = (
                defaultdict(list)
            )

            for result in results:

                scenario_groups[
                    result["scenario"]
                ].append(
                    result
                )

            output[
                group_name
            ] = {
                "action_mode":
                    self.action_mode,

                "overall":
                    self.aggregate(
                        results
                    ),

                "scenarios": {
                    scenario_name:
                        self.aggregate(
                            scenario_results
                        )

                    for (
                        scenario_name,
                        scenario_results,
                    )
                    in
                    scenario_groups.items()
                },
            }

        return output

    # ========================================================
    # Save CSV
    # ========================================================

    @staticmethod
    def save_csv(
        results,
        path,
    ):

        if not results:
            return

        directory = (
            os.path.dirname(
                path
            )
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        with open(
            path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = (
                csv.DictWriter(
                    f,
                    fieldnames=
                        list(
                            results[
                                0
                            ].keys()
                        ),
                )
            )

            writer.writeheader()

            writer.writerows(
                results
            )

    # ========================================================
    # Save JSON
    # ========================================================

    @staticmethod
    def save_json(
        data,
        path,
    ):

        directory = (
            os.path.dirname(
                path
            )
        )

        if directory:
            os.makedirs(
                directory,
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