from torch.distributions import Categorical

import argparse

import numpy as np
import torch

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

from .ppo import PPO


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


def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--checkpoint",

        type=str,

        default=(
            "checkpoints/"
            "ppo_final.pt"
        ),
    )

    parser.add_argument(
        "--stage",

        type=int,

        default=3,
    )

    parser.add_argument(
        "--spawn-level",

        type=int,

        default=2,
    )

    parser.add_argument(
        "--reward-mode",

        type=str,

        default=(
            "target_only"
        ),

        choices=[
            "target_only",
            "obstacle_shaping",
        ],
    )

    parser.add_argument(
        "--episodes",

        type=int,

        default=10,
    )

    parser.add_argument(
        "--render",

        action="store_true",
    )

    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    render_mode = (
        "human"
        if args.render
        else None
    )

    env = (
        CooperativeTransportEnv(
            num_agents=4,

            curriculum_stage=
                args.stage,

            spawn_level=
                args.spawn_level,

            reward_mode=
                args.reward_mode,

            render_mode=
                render_mode,
        )
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

        device=device,
    )

    ppo.load(
        args.checkpoint,

        load_optimizer=False,
    )

    ppo.network.eval()

    successes = 0

    lengths = []

    returns = []

    target_moves = []

    obstacle_moves = []

    with torch.no_grad():

        for episode in range(
            args.episodes
        ):

            episode_seed = 10000 + episode

            np.random.seed(
                episode_seed
            )

            torch.manual_seed(
                episode_seed
            )

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(
                    episode_seed
                )

            observations, _ = env.reset(
                seed=episode_seed
            )

            episode_return = 0.0

            episode_length = 0

            episode_target_moves = 0

            episode_obstacle_moves = 0

            success = False

            while env.agents:

                current_agents = (
                    env.agents[:]
                )

                actions = {}

                for agent in current_agents:

                    flat = flatten_obs(
                        env.observation_space(agent),
                        observations[agent],
                    )

                    obs_tensor = (
                        torch.tensor(
                            flat,
                            dtype=torch.float32,
                            device=device,
                        )
                        .unsqueeze(0)
                    )

                    logits, _ = ppo.network(
                        obs_tensor
                    )

                    # 기존:
                    # action = int(
                    #     torch.argmax(
                    #         logits,
                    #         dim=-1,
                    #     ).item()
                    # )

                    # 수정: 학습 때처럼 policy distribution에서 sampling
                    distribution = Categorical(
                        logits=logits
                    )

                    action = int(
                        distribution.sample().item()
                    )

                    actions[agent] = np.array(
                        [
                            action,
                            NO_MESSAGE,
                        ],
                        dtype=np.int64,
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
                    _,
                    _,
                    infos,
                ) = env.step(
                    actions
                )

                first_agent = (
                    current_agents[0]
                )

                info = (
                    infos[
                        first_agent
                    ]
                )

                episode_return += (
                    rewards[
                        first_agent
                    ]
                )

                episode_length += 1

                episode_target_moves += (
                    int(
                        info[
                            "target_moved"
                        ]
                    )
                )

                episode_obstacle_moves += (
                    info[
                        "obstacle_moves"
                    ]
                )

                success = (
                    info["success"]
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

            target_moves.append(
                episode_target_moves
            )

            obstacle_moves.append(
                episode_obstacle_moves
            )

            print(
                f"Episode "
                f"{episode + 1}: "
                f"success="
                f"{success}, "
                f"length="
                f"{episode_length}, "
                f"return="
                f"{episode_return:.3f}, "
                f"target_moves="
                f"{episode_target_moves}, "
                f"obstacle_moves="
                f"{episode_obstacle_moves}"
            )

    print()
    print(
        "=" * 50
    )

    print(
        f"Success rate: "
        f"{successes / args.episodes:.3f}"
    )

    print(
        f"Average length: "
        f"{np.mean(lengths):.2f}"
    )

    print(
        f"Average return: "
        f"{np.mean(returns):.3f}"
    )

    print(
        f"Average target moves: "
        f"{np.mean(target_moves):.2f}"
    )

    print(
        f"Average obstacle moves: "
        f"{np.mean(obstacle_moves):.2f}"
    )


if __name__ == "__main__":
    main()