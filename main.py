from env import (
    CooperativeTransportEnv,
)


def main():

    env = CooperativeTransportEnv(
        width=12,
        height=12,
        num_agents=4,
        vision_size=5,
        max_steps=200,
        render_mode="human",
    )

    observations, infos = (
        env.reset(seed=42)
    )

    while env.agents:

        actions = {
            agent:
                env.action_space(
                    agent
                ).sample()

            for agent in env.agents
        }

        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = env.step(actions)

    env.close()


if __name__ == "__main__":
    main()