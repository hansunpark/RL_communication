import os

from collections import deque

import numpy as np
import torch

from gymnasium.spaces import (
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

from ...ppo import PPO

from ...rollout_buffer import (
    MultiAgentRolloutBuffer,
)

from ..common import (
    set_seed,
    flatten_obs,
)

from .curriculum import (
    Stage7ScenarioSampler,
    Stage7CurriculumManager,
)


# ==========================================================
# Configuration
#
# Stage 0~6는 이미 stages/stage_0_6/train.py로 완료했고, 그
# 결과물인 stage_6_mastered.pt에서 이어서 Stage 7만
# fine-tuning한다.
#
# 이 파일은 Stage0~6 curriculum과는 완전히 분리되어 있다.
# 같은 학습 스크립트를 직접 수정하면 두 curriculum의 상태
# 머신이 같은 episode 루프 안에서 충돌하기 때문이다.
# ==========================================================

SEED = 42

STAGE7_MAX_EPISODES = 10000

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

STAGE7_SUCCESS_THRESHOLD = 0.85

STAGE7_WINDOW_SIZE = 100

STAGE7_MIN_EPISODES = 200

STAGE7_ORIGINAL_PROBABILITY = 0.40

STAGE7_START_CHECKPOINT = (
    "checkpoints/stage_6_mastered.pt"
)


# ==========================================================
# Utility
# ==========================================================

def scenario_label(scenario):

    if scenario is None:
        return "original(x=4)"

    return (
        "shifted(x="
        f"{scenario['target']['x']})"
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
        "stage7"
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

    stage7_sampler = (
        Stage7ScenarioSampler(
            map_width=env.width,

            original_probability=
                STAGE7_ORIGINAL_PROBABILITY,
        )
    )

    stage7_curriculum = (
        Stage7CurriculumManager(
            success_threshold=
                STAGE7_SUCCESS_THRESHOLD,

            window_size=
                STAGE7_WINDOW_SIZE,

            min_episodes=
                STAGE7_MIN_EPISODES,
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

    print(
        "Loading Stage 6 checkpoint: "
        f"{STAGE7_START_CHECKPOINT}"
    )

    ppo.load(
        STAGE7_START_CHECKPOINT,
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
        STAGE7_MAX_EPISODES + 1,
    ):

        scenario = (
            stage7_sampler.sample()
        )

        episode_seed = (
            SEED + episode
        )

        if scenario is None:

            # 기존 Stage 6 문제 그대로
            env.set_curriculum(
                stage=6,
                spawn_level=2,
                reward_mode=
                    "obstacle_shaping",
            )

            observations, infos = (
                env.reset(
                    seed=episode_seed
                )
            )

        else:

            # Stage 7: target/goal/obstacle의
            # x 위치만 바뀐 문제
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

        stage7_curriculum.record(
            episode_success
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
            "stage7/substage",
            stage7_sampler.level,
            episode,
        )

        writer.add_scalar(
            "stage7/success_rate",
            stage7_curriculum
            .success_rate,
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
                f"{stage7_sampler.name:<20} "
                f"scenario="
                f"{scenario_label(scenario):<16} "
                f"success100="
                f"{stage7_curriculum.success_rate:.3f} "
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
        # Periodic checkpoint
        # ==================================================

        if (
            episode
            % SAVE_INTERVAL
            == 0
        ):

            ppo.save(
                "checkpoints/"
                f"stage7_episode_"
                f"{episode}.pt"
            )

        # ==================================================
        # Stage 7 substage transition
        # ==================================================

        if stage7_curriculum.mastered():

            current_name = (
                stage7_sampler.name
            )

            print()
            print(
                "=" * 70
            )

            print(
                f"{current_name} mastered! "
                f"success rate="
                f"{stage7_curriculum.success_rate:.3f}"
            )

            print(
                "=" * 70
            )

            print()

            checkpoint_path = (
                "checkpoints/"
                f"{current_name}"
                "_mastered.pt"
            )

            ppo.save(
                checkpoint_path
            )

            advanced = (
                stage7_sampler.advance()
            )

            # 이전 substage의 rollout을
            # 다음 substage와 섞지 않는다.
            buffer.reset()

            length_window.clear()

            return_window.clear()

            target_move_window.clear()

            obstacle_move_window.clear()

            if advanced:

                stage7_curriculum.reset_level()

                print(
                    f"Moving to "
                    f"{stage7_sampler.name}"
                )

            else:

                ppo.save(
                    "checkpoints/"
                    "stage_7_mastered.pt"
                )

                stage7_curriculum.completed = True

                print(
                    "Stage 7 curriculum "
                    "completed!"
                )

                break

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
        "stage7_final.pt"
    )

    writer.close()

    env.close()

    print(
        "Stage 7 training complete."
    )


if __name__ == "__main__":
    main()
