import os
import random

from collections import deque

import numpy as np
import torch

from gymnasium.spaces import (
    flatten,
    flatdim,
)

from torch.utils.tensorboard import (
    SummaryWriter,
)

from env import (
    CooperativeTransportEnv,
)

from env.constants import (
    NO_MESSAGE,
)

from .ppo import PPO

from .rollout_buffer import (
    MultiAgentRolloutBuffer,
)


SEED = 42

TOTAL_EPISODES = 10000

ROLLOUT_SIZE = 4096

GAMMA = 0.99
GAE_LAMBDA = 0.95

LEARNING_RATE = 3e-4

HIDDEN_DIM = 256

ENTROPY_COEF = 0.03

VALUE_COEF = 0.5

CLIP_COEF = 0.2

UPDATE_EPOCHS = 4

MINIBATCH_SIZE = 256

SUCCESS_THRESHOLD = 0.80

MIN_EPISODES_PER_STAGE = 200

SAVE_INTERVAL = 200


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
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


def main():

    set_seed(SEED)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    os.makedirs(
        "checkpoints",
        exist_ok=True,
    )

    writer = SummaryWriter(
        "runs/"
        "curriculum_ppo"
    )

    env = CooperativeTransportEnv(
        num_agents=4,
        vision_size=5,
        max_steps=200,

        curriculum_stage=0,

        step_penalty=-0.01,

        distance_reward_coef=0.2,

        success_reward=10.0,
    )

    sample_agent = (
        env.possible_agents[0]
    )

    obs_dim = flatdim(
        env.observation_space(
            sample_agent
        )
    )

    print(
        "Observation dim:",
        obs_dim,
    )

    ppo = PPO(
        obs_dim=obs_dim,
        num_actions=5,

        learning_rate=
            LEARNING_RATE,

        hidden_dim=
            HIDDEN_DIM,

        clip_coef=
            CLIP_COEF,

        value_coef=
            VALUE_COEF,

        entropy_coef=
            ENTROPY_COEF,

        update_epochs=
            UPDATE_EPOCHS,

        minibatch_size=
            MINIBATCH_SIZE,

        device=device,
    )

    buffer = (
        MultiAgentRolloutBuffer(
            device=device
        )
    )

    current_stage = 0

    stage_episode_count = 0

    success_window = deque(
        maxlen=100
    )

    length_window = deque(
        maxlen=100
    )

    return_window = deque(
        maxlen=100
    )

    global_step = 0

    for episode in range(
        1,
        TOTAL_EPISODES + 1,
    ):

        env.set_curriculum_stage(
            current_stage
        )

        observations, _ = (
            env.reset(
                seed=
                    SEED
                    + episode
            )
        )

        episode_return = 0.0

        episode_length = 0

        episode_success = False

        while env.agents:

            current_agents = (
                env.agents[:]
            )

            flat_observations = {}

            transition_data = {}

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

                (
                    action,
                    log_prob,
                    value,
                ) = ppo.act(
                    flat
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

                flat_observations[
                    agent
                ] = flat

                transition_data[
                    agent
                ] = (
                    action,
                    log_prob,
                    value,
                )

            (
                next_observations,
                rewards,
                terminations,
                truncations,
                infos,
            ) = env.step(
                actions
            )

            for agent in (
                current_agents
            ):

                done = (
                    terminations[
                        agent
                    ]
                    or
                    truncations[
                        agent
                    ]
                )

                if done:

                    next_value = 0.0

                else:

                    next_flat = (
                        flatten_obs(
                            env.observation_space(
                                agent
                            ),
                            next_observations[
                                agent
                            ],
                        )
                    )

                    next_value = (
                        ppo.value(
                            next_flat
                        )
                    )

                (
                    action,
                    log_prob,
                    value,
                ) = (
                    transition_data[
                        agent
                    ]
                )

                buffer.add(
                    agent=agent,

                    observation=
                        flat_observations[
                            agent
                        ],

                    action=action,

                    log_prob=
                        log_prob,

                    reward=
                        rewards[
                            agent
                        ],

                    done=done,

                    value=value,

                    next_value=
                        next_value,
                )

                global_step += 1

            episode_return += (
                rewards[
                    current_agents[
                        0
                    ]
                ]
            )

            episode_length += 1

            episode_success = (
                infos[
                    current_agents[
                        0
                    ]
                ][
                    "success"
                ]
            )

            observations = (
                next_observations
            )

            if (
                len(buffer)
                >= ROLLOUT_SIZE
            ):

                metrics = (
                    ppo.update(
                        buffer,
                        gamma=GAMMA,
                        gae_lambda=
                            GAE_LAMBDA,
                    )
                )

                writer.add_scalar(
                    "loss/policy",
                    metrics[
                        "policy_loss"
                    ],
                    global_step,
                )

                writer.add_scalar(
                    "loss/value",
                    metrics[
                        "value_loss"
                    ],
                    global_step,
                )

                writer.add_scalar(
                    "policy/entropy",
                    metrics[
                        "entropy"
                    ],
                    global_step,
                )

                buffer.reset()

        stage_episode_count += 1

        success_window.append(
            int(
                episode_success
            )
        )

        length_window.append(
            episode_length
        )

        return_window.append(
            episode_return
        )

        success_rate = (
            np.mean(
                success_window
            )
        )

        avg_length = np.mean(
            length_window
        )

        avg_return = np.mean(
            return_window
        )

        writer.add_scalar(
            "episode/return",
            episode_return,
            episode,
        )

        writer.add_scalar(
            "episode/length",
            episode_length,
            episode,
        )

        writer.add_scalar(
            "episode/success",
            int(
                episode_success
            ),
            episode,
        )

        writer.add_scalar(
            "curriculum/stage",
            current_stage,
            episode,
        )

        writer.add_scalar(
            "curriculum/"
            "success_rate_100",
            success_rate,
            episode,
        )

        if episode % 10 == 0:

            print(
                f"[Episode "
                f"{episode:5d}] "
                f"Stage={current_stage} "
                f"success100="
                f"{success_rate:.3f} "
                f"avg_len="
                f"{avg_length:.1f} "
                f"avg_return="
                f"{avg_return:.3f}"
            )

        # ================================================
        # Curriculum promotion
        # ================================================

        enough_episodes = (
            stage_episode_count
            >= MIN_EPISODES_PER_STAGE
        )

        enough_samples = (
            len(success_window)
            >= 100
        )

        mastered = (
            success_rate
            >= SUCCESS_THRESHOLD
        )

        if (
            current_stage < 4
            and enough_episodes
            and enough_samples
            and mastered
        ):

            old_stage = (
                current_stage
            )

            current_stage += 1

            stage_episode_count = 0

            success_window.clear()
            length_window.clear()
            return_window.clear()

            print()
            print(
                "================================"
            )

            print(
                f"Stage {old_stage} mastered."
            )

            print(
                f"Moving to Stage "
                f"{current_stage}"
            )

            print(
                "================================"
            )

            print()

            ppo.save(
                "checkpoints/"
                f"stage_"
                f"{old_stage}"
                f"_complete.pt"
            )

        if (
            episode
            % SAVE_INTERVAL
            == 0
        ):

            ppo.save(
                "checkpoints/"
                f"episode_"
                f"{episode}.pt"
            )

    if len(buffer) > 0:

        ppo.update(
            buffer,
            gamma=GAMMA,
            gae_lambda=
                GAE_LAMBDA,
        )

    ppo.save(
        "checkpoints/"
        "ppo_final.pt"
    )

    writer.close()

    env.close()

    print(
        "Training complete."
    )


if __name__ == "__main__":
    main()