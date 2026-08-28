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

from .curriculum import (
    CurriculumManager,
)


# ==========================================================
# Configuration
# ==========================================================

SEED = 42

TOTAL_EPISODES = 20000

ROLLOUT_SIZE = 4096

GAMMA = 0.99

GAE_LAMBDA = 0.95

LEARNING_RATE = 3e-4

HIDDEN_DIM = 256

CLIP_COEF = 0.2

VALUE_COEF = 0.5

ENTROPY_COEF = 0.03

UPDATE_EPOCHS = 4

MINIBATCH_SIZE = 256

SUCCESS_THRESHOLD = 0.85

CURRICULUM_WINDOW = 100

MIN_EPISODES_PER_UNIT = 200

STAGE4_BASELINE_PATIENCE = 600

SAVE_INTERVAL = 200

STOP_WHEN_MASTERED = True


# ==========================================================
# Utility
# ==========================================================

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


# ==========================================================
# Main
# ==========================================================

def main():

    set_seed(SEED)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    os.makedirs(
        "checkpoints",
        exist_ok=True,
    )

    writer = SummaryWriter(
        "runs/"
        "curriculum_v2"
    )

    curriculum = (
        CurriculumManager(
            success_threshold=
                SUCCESS_THRESHOLD,

            window_size=
                CURRICULUM_WINDOW,

            min_episodes=
                MIN_EPISODES_PER_UNIT,

            stage4_baseline_patience=
                STAGE4_BASELINE_PATIENCE,
        )
    )

    config = (
        curriculum.config()
    )

    env = (
        CooperativeTransportEnv(
            num_agents=4,

            vision_size=5,

            max_steps=200,

            curriculum_stage=
                config["stage"],

            spawn_level=
                config[
                    "spawn_level"
                ],

            reward_mode=
                config[
                    "reward_mode"
                ],

            step_penalty=-0.01,

            distance_reward_coef=
                0.2,

            obstacle_reward_coef=
                0.1,

            success_reward=10.0,
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

    print(
        f"Observation dim: "
        f"{obs_dim}"
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

    length_window = deque(
        maxlen=100
    )

    return_window = deque(
        maxlen=100
    )

    target_move_window = deque(
        maxlen=100
    )

    obstacle_move_window = deque(
        maxlen=100
    )

    global_step = 0

    last_metrics = None

    for episode in range(
        1,
        TOTAL_EPISODES + 1,
    ):

        config = (
            curriculum.config()
        )

        env.set_curriculum(
            stage=
                config["stage"],

            spawn_level=
                config[
                    "spawn_level"
                ],

            reward_mode=
                config[
                    "reward_mode"
                ],
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

        episode_target_moves = 0

        episode_obstacle_moves = 0

        while env.agents:

            current_agents = (
                env.agents[:]
            )

            flat_observations = {}

            transition_data = {}

            actions = {}

            # ============================================
            # Policy actions
            # ============================================

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
                ) = (
                    ppo.act(
                        flat
                    )
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

            # ============================================
            # Environment step
            # ============================================

            (
                next_observations,
                rewards,
                terminations,
                truncations,
                infos,
            ) = env.step(
                actions
            )

            # ============================================
            # Store transitions
            # ============================================

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

            first_agent = (
                current_agents[0]
            )

            team_reward = (
                rewards[
                    first_agent
                ]
            )

            info = (
                infos[
                    first_agent
                ]
            )

            episode_return += (
                team_reward
            )

            episode_length += 1

            episode_success = (
                info["success"]
            )

            episode_target_moves += int(
                info[
                    "target_moved"
                ]
            )

            episode_obstacle_moves += (
                info[
                    "obstacle_moves"
                ]
            )

            observations = (
                next_observations
            )

            # ============================================
            # PPO update
            # ============================================

            if (
                len(buffer)
                >= ROLLOUT_SIZE
            ):

                last_metrics = (
                    ppo.update(
                        buffer,

                        gamma=GAMMA,

                        gae_lambda=
                            GAE_LAMBDA,
                    )
                )

                writer.add_scalar(
                    "loss/policy",

                    last_metrics[
                        "policy_loss"
                    ],

                    global_step,
                )

                writer.add_scalar(
                    "loss/value",

                    last_metrics[
                        "value_loss"
                    ],

                    global_step,
                )

                writer.add_scalar(
                    "policy/entropy",

                    last_metrics[
                        "entropy"
                    ],

                    global_step,
                )

                buffer.reset()

        # ==================================================
        # Episode statistics
        # ==================================================

        length_window.append(
            episode_length
        )

        return_window.append(
            episode_return
        )

        target_move_window.append(
            episode_target_moves
        )

        obstacle_move_window.append(
            episode_obstacle_moves
        )

        curriculum_event = (
            curriculum.record_episode(
                episode_success
            )
        )

        success_rate = (
            curriculum.success_rate
        )

        avg_length = (
            np.mean(
                length_window
            )
        )

        avg_return = (
            np.mean(
                return_window
            )
        )

        avg_target_moves = (
            np.mean(
                target_move_window
            )
        )

        avg_obstacle_moves = (
            np.mean(
                obstacle_move_window
            )
        )

        config = (
            curriculum.config()
        )

        # ==================================================
        # TensorBoard
        # ==================================================

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
            config["stage"],
            episode,
        )

        writer.add_scalar(
            "curriculum/spawn_level",
            config[
                "spawn_level"
            ],
            episode,
        )

        writer.add_scalar(
            "curriculum/success_rate",
            success_rate,
            episode,
        )

        writer.add_scalar(
            "diagnostics/"
            "target_moves",

            episode_target_moves,

            episode,
        )

        writer.add_scalar(
            "diagnostics/"
            "obstacle_moves",

            episode_obstacle_moves,

            episode,
        )

        # ==================================================
        # Console
        # ==================================================

        if episode % 10 == 0:

            print(
                f"[Episode "
                f"{episode:5d}] "
                f"{curriculum.unit_name:<28} "
                f"reward="
                f"{curriculum.reward_mode:<17} "
                f"success100="
                f"{success_rate:.3f} "
                f"avg_len="
                f"{avg_length:.1f} "
                f"return="
                f"{avg_return:.3f} "
                f"Tmove="
                f"{avg_target_moves:.2f} "
                f"Omove="
                f"{avg_obstacle_moves:.2f}"
            )

        # ==================================================
        # Curriculum transition
        # ==================================================

        if curriculum_event:

            print()
            print(
                "=" * 70
            )

            print(
                curriculum_event[
                    "message"
                ]
            )

            print(
                "=" * 70
            )

            print()

            checkpoint_path = (
                "checkpoints/"
                +
                curriculum_event[
                    "checkpoint"
                ]
            )

            ppo.save(
                checkpoint_path
            )

            # Important:
            # Do not mix samples from
            # different curriculum distributions.
            buffer.reset()

            length_window.clear()

            return_window.clear()

            target_move_window.clear()

            obstacle_move_window.clear()

            if (
                curriculum_event[
                    "type"
                ]
                == "completed"
                and
                STOP_WHEN_MASTERED
            ):

                print(
                    "Final curriculum "
                    "mastered."
                )

                break

        # ==================================================
        # Periodic checkpoint
        # ==================================================

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

    # ======================================================
    # Final update
    # ======================================================

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