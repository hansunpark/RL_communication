import json

import numpy as np
import torch

from gymnasium.spaces import flatdim

from env import CooperativeTransportEnv
from env.constants import NO_MESSAGE

from ...ppo import PPO
from ..common import flatten_obs
from .curriculum import Stage7ScenarioSampler


# ==========================================================
# Stage 7 lane별 성공률 진단
#
# Stage7CurriculumManager는 7A/7B/7C 전체를 섞은 평균
# success rate만 보고 mastery를 판정했기 때문에, 특정 x
# lane 하나가 완전히 실패해도 다른 lane들이 잘 되면 평균이
# 85%를 넘어 통과할 수 있다 (실제로 lane_left/x=2에서 이런
# 일이 발생한 것을 OOD benchmark에서 확인했다).
#
# 이 스크립트는 stage_7_mastered.pt를 고정한 채, 7C가 다루는
# 모든 x lane 각각에 대해 별도로 success rate를 측정해서
# 어느 lane이 실제로 취약한지 데이터로 확인한다.
# ==========================================================

CHECKPOINT = "checkpoints/stage_7_mastered.pt"

EPISODES_PER_LANE = 50

SEED_BASE = 200000

OUTPUT_PATH = (
    "evaluation_results/"
    "stage7_lane_diagnostic.json"
)


def run_lane(
    env,
    ppo,
    sampler,
    lane_x,
    episodes,
    seed_start,
):

    successes = 0

    lengths = []

    returns = []

    obstacle_moves = []

    scenario = (
        sampler.make_scenario(
            lane_x
        )
    )

    for episode in range(
        episodes
    ):

        seed = (
            seed_start
            + episode
        )

        observations, infos = (
            env.reset(
                seed=seed,
                options={
                    "scenario":
                        scenario
                },
            )
        )

        episode_return = 0.0

        episode_length = 0

        episode_obstacle_moves = 0

        success = False

        while env.agents:

            current_agents = (
                env.agents[:]
            )

            actions = {}

            for agent in (
                current_agents
            ):

                flat = flatten_obs(
                    env.observation_space(
                        agent
                    ),

                    observations[
                        agent
                    ],
                )

                action, _, _ = (
                    ppo.act(flat)
                )

                actions[
                    agent
                ] = np.array(
                    [
                        action,
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
            ) = env.step(
                actions
            )

            first_agent = (
                current_agents[0]
            )

            episode_return += (
                rewards[
                    first_agent
                ]
            )

            episode_length += 1

            episode_obstacle_moves += (
                infos[
                    first_agent
                ][
                    "obstacle_moves"
                ]
            )

            success = (
                infos[
                    first_agent
                ]["success"]
            )

        successes += int(
            success
        )

        lengths.append(
            episode_length
        )

        returns.append(
            episode_return
        )

        obstacle_moves.append(
            episode_obstacle_moves
        )

    return {
        "lane_x": lane_x,

        "episodes": episodes,

        "success_rate":
            successes
            / episodes,

        "avg_length":
            float(
                np.mean(lengths)
            ),

        "avg_return":
            float(
                np.mean(returns)
            ),

        "avg_obstacle_moves":
            float(
                np.mean(
                    obstacle_moves
                )
            ),
    }


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    env = (
        CooperativeTransportEnv(
            num_agents=4,

            vision_size=5,

            max_steps=200,

            curriculum_stage=6,

            spawn_level=2,

            reward_mode=
                "obstacle_shaping",

            step_penalty=-0.01,

            distance_reward_coef=
                0.2,

            obstacle_reward_coef=
                0.1,

            success_reward=10.0,
        )
    )

    sampler = (
        Stage7ScenarioSampler(
            map_width=env.width
        )
    )

    # 7C가 다루는 모든 lane
    sampler.level = 2

    lanes = (
        sampler.allowed_x_positions()
    )

    print(
        f"Lanes to test: {lanes}"
    )

    sample_agent = (
        env.possible_agents[0]
    )

    obs_dim = flatdim(
        env.observation_space(
            sample_agent
        )
    )

    ppo = PPO(
        obs_dim=obs_dim,

        num_actions=5,

        device=device,
    )

    print(
        f"Loading checkpoint: "
        f"{CHECKPOINT}"
    )

    ppo.load(
        CHECKPOINT,
        load_optimizer=False,
    )

    ppo.network.eval()

    results = []

    for lane_x in lanes:

        seed_start = (
            SEED_BASE
            +
            lane_x
            * 10000
        )

        result = run_lane(
            env=env,
            ppo=ppo,
            sampler=sampler,
            lane_x=lane_x,
            episodes=
                EPISODES_PER_LANE,
            seed_start=
                seed_start,
        )

        results.append(result)

        print(
            f"x={lane_x:>2}  "
            f"success="
            f"{result['success_rate']:.3f}  "
            f"avg_len="
            f"{result['avg_length']:.1f}  "
            f"avg_return="
            f"{result['avg_return']:.3f}  "
            f"avg_obs_moves="
            f"{result['avg_obstacle_moves']:.2f}"
        )

    with open(
        OUTPUT_PATH,
        "w",
    ) as f:

        json.dump(
            {
                "checkpoint":
                    CHECKPOINT,

                "episodes_per_lane":
                    EPISODES_PER_LANE,

                "results": results,
            },

            f,

            indent=2,
        )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )

    env.close()


if __name__ == "__main__":
    main()
