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
        default=0,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    env = CooperativeTransportEnv(
        num_agents=4,

        curriculum_stage=
            args.stage,

        render_mode=
            "human",
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
        args.checkpoint
    )

    ppo.network.eval()

    successes = 0

    with torch.no_grad():

        for episode in range(
            args.episodes
        ):

            observations, _ = (
                env.reset(
                    seed=
                        10000
                        + episode
                )
            )

            success = False

            while env.agents:

                actions = {}

                for agent in (
                    env.agents
                ):

                    flat = flatten_obs(
                        env.observation_space(
                            agent
                        ),
                        observations[
                            agent
                        ],
                    )

                    obs_tensor = (
                        torch.tensor(
                            flat,
                            dtype=
                                torch.float32,
                            device=
                                device,
                        ).unsqueeze(0)
                    )

                    logits, _ = (
                        ppo.network(
                            obs_tensor
                        )
                    )

                    action = int(
                        torch.argmax(
                            logits,
                            dim=-1,
                        ).item()
                    )

                    actions[
                        agent
                    ] = np.array(
                        [
                            action,
                            NO_MESSAGE,
                        ],
                        dtype=
                            np.int64,
                    )

                (
                    observations,
                    _,
                    _,
                    _,
                    infos,
                ) = env.step(
                    actions
                )

                if infos:

                    success = (
                        next(
                            iter(
                                infos.values()
                            )
                        )[
                            "success"
                        ]
                    )

            successes += int(
                success
            )

    print(
        "Success rate:",
        successes
        / args.episodes,
    )


if __name__ == "__main__":
    main()