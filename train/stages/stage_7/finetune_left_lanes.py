import os

from collections import deque

import numpy as np
import torch

from gymnasium.spaces import flatdim

from torch.utils.tensorboard import (
    SummaryWriter,
)

from env import CooperativeTransportEnv
from env.constants import NO_MESSAGE

from ...ppo import PPO
from ...rollout_buffer import (
    MultiAgentRolloutBuffer,
)
from ..common import (
    set_seed,
    flatten_obs,
)
from .curriculum import (
    LeftLaneFocusSampler,
    LeftLaneFocusCurriculumManager,
)


# ==========================================================
# Stage 7 left-lane 집중 fine-tuning
#
# lane_diagnostic.py로 stage_7_mastered.pt를 진단한 결과:
#   x=1 -> success 0.00
#   x=2 -> success 0.00
#   x=3 -> success 0.72
#   x=4~8 -> success 0.90~1.00
#
# Stage7CurriculumManager는 전체 평균만 보고 mastery를
# 판정했기 때문에 이 실패를 놓쳤다. 여기서는 같은 실수를
# 반복하지 않도록 LeftLaneFocusCurriculumManager가 취약한
# lane(x=1,2,3)과 나머지(retention)를 "따로" 추적해서 전부
# 개별 기준을 넘겨야 끝난다.
#
# stage_7_mastered.pt에서 이어서 시작한다. Stage7 curriculum
# 자체를 다시 돌리지 않는다.
# ==========================================================

SEED = 42

MAX_EPISODES = 6000

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

SAVE_INTERVAL = 200

START_CHECKPOINT = (
    "checkpoints/stage_7_mastered.pt"
)


def main():

    set_seed(SEED)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    os.makedirs(
        "checkpoints",
        exist_ok=True,
    )

    writer = SummaryWriter(
        "runs/stage7_left_focus"
    )

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
        LeftLaneFocusSampler(
            map_width=env.width
        )
    )

    curriculum = (
        LeftLaneFocusCurriculumManager()
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
        hidden_dim=HIDDEN_DIM,
        clip_coef=CLIP_COEF,
        value_coef=VALUE_COEF,
        entropy_coef=
            ENTROPY_COEF,
        update_epochs=
            UPDATE_EPOCHS,
        minibatch_size=
            MINIBATCH_SIZE,
        device=device,
    )

    print(
        "Loading checkpoint: "
        f"{START_CHECKPOINT}"
    )

    ppo.load(
        START_CHECKPOINT,
        load_optimizer=True,
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

    global_step = 0

    for episode in range(
        1,
        MAX_EPISODES + 1,
    ):

        lane_x, scenario = (
            sampler.sample()
        )

        episode_seed = (
            SEED + episode
        )

        observations, infos = (
            env.reset(
                seed=episode_seed,
                options={
                    "scenario":
                        scenario
                },
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
                ) = ppo.act(flat)

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
            ) = env.step(actions)

            for agent in (
                current_agents
            ):

                done = (
                    terminations[
                        agent
                    ]
                    or truncations[
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
                ) = transition_data[
                    agent
                ]

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

            episode_return += (
                rewards[
                    first_agent
                ]
            )

            episode_length += 1

            episode_success = (
                infos[
                    first_agent
                ]["success"]
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

                buffer.reset()

        length_window.append(
            episode_length
        )

        return_window.append(
            episode_return
        )

        curriculum.record(
            lane_x,
            episode_success,
        )

        writer.add_scalar(
            "episode/return",
            episode_return,
            episode,
        )

        writer.add_scalar(
            "episode/success",
            int(episode_success),
            episode,
        )

        writer.add_scalar(
            f"lane/x{lane_x}_success",
            int(episode_success),
            episode,
        )

        if episode % 10 == 0:

            status = (
                curriculum.status()
            )

            status_str = "  ".join(
                f"{bucket}="
                f"{info['success_rate']:.2f}"
                f"({info['n']})"

                for bucket, info
                in status.items()
            )

            print(
                f"[Episode "
                f"{episode:5d}] "
                f"lane=x{lane_x} "
                f"success={int(episode_success)} "
                f"len={episode_length:3d}  |  "
                f"{status_str}"
            )

        if (
            episode % SAVE_INTERVAL
            == 0
        ):

            ppo.save(
                "checkpoints/"
                f"stage7_left_focus_"
                f"episode_{episode}.pt"
            )

        if curriculum.mastered():

            print()
            print("=" * 70)

            print(
                "Left-lane focus "
                "fine-tuning mastered!"
            )

            print(
                curriculum.status()
            )

            print("=" * 70)

            ppo.save(
                "checkpoints/"
                "stage_7_left_focus_"
                "mastered.pt"
            )

            curriculum.completed = True

            break

    if len(buffer) > 0:

        ppo.update(
            buffer,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
        )

    ppo.save(
        "checkpoints/"
        "stage7_left_focus_final.pt"
    )

    writer.close()

    env.close()

    print(
        "Left-lane focus "
        "fine-tuning complete."
    )

    if not curriculum.completed:

        print(
            "Warning: reached "
            "MAX_EPISODES without "
            "hitting all bucket "
            "thresholds."
        )

        print(
            curriculum.status()
        )


if __name__ == "__main__":
    main()
