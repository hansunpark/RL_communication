import numpy as np

from env import (
    CooperativeTransportEnv,
)


def main():

    env = (
        CooperativeTransportEnv(
            curriculum_stage=3,

            spawn_level=2,

            render_mode="human",
        )
    )

    observations, infos = (
        env.reset(
            seed=42
        )
    )

    while env.agents:

        actions = {
            agent:
                env.action_space(
                    agent
                ).sample()

            for agent
            in env.agents
        }

        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = env.step(
            actions
        )

    env.close()


if __name__ == "__main__":
    main()