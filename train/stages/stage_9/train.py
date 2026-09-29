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
    HELP,
)

from ...ppo import PPO

from ...rollout_buffer import (
    MultiAgentRolloutBuffer,
)

from ..common import (
    set_seed,
    flatten_obs,
)

from ..stage_7.curriculum import (
    Stage7CurriculumManager,
)

from .curriculum import (
    Stage9LevelSampler,
)


# ==========================================================
# Configuration
#
# 이 스크립트는 "통신이 실제로 필요해지는지"를 검증하지
# 않는다 - target 위치가 Stage 0-6 기본값으로 고정되어 있어서
# 멀리 스폰된 에이전트도 이미 어디로 가야 하는지 학습으로
# 알고 있을 수 있다. 여기서는 오직 파이프라인(통신 head를
# 포함한 결합 액션 PPO 학습, HELP 페널티, split-spawn)이
# 기계적으로 정상 동작하는지만 확인한다.
#
# 진짜 필요성 검증은 Stage 8의 위치 일반화(8B1~)가 끝난 뒤,
# 이 스폰 방식을 랜덤 위치 시나리오 위에 다시 얹어서 해야
# 한다. docs/curriculum_plan.md 참고.
# ==========================================================

SEED = 42

STAGE9_MAX_EPISODES = 20000

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

STAGE9_SUCCESS_THRESHOLD = 0.85

STAGE9_WINDOW_SIZE = 100

STAGE9_MIN_EPISODES = 200

STAGE9_START_CHECKPOINT = (
    "checkpoints/"
    "stage_8A_size_mastered.pt"
)

STAGE9_NEAR_RADIUS = 2

STAGE9_FAR_MIN_DISTANCE = 6

# HELP를 누를 때 내는 개별 페널티. 지금은 일부러 작게
# 잡는다 (env.py 기본값과 동일) - 연구가 진행되면서 점차
# 올려볼 예정.
STAGE9_HELP_PENALTY = 0.002


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
        "stage9"
    )

    env = (
        CooperativeTransportEnv(
            num_agents=4,

            vision_size=5,

            max_steps=200,

            curriculum_stage=0,

            spawn_level=0,

            reward_mode=
                "target_only",

            step_penalty=-0.01,

            distance_reward_coef=
                0.2,

            obstacle_reward_coef=
                0.1,

            success_reward=10.0,

            help_penalty=
                STAGE9_HELP_PENALTY,
        )
    )

    stage9_sampler = (
        Stage9LevelSampler(
            near_radius=
                STAGE9_NEAR_RADIUS,

            far_min_distance=
                STAGE9_FAR_MIN_DISTANCE,
        )
    )

    stage9_curriculum = (
        Stage7CurriculumManager(
            success_threshold=
                STAGE9_SUCCESS_THRESHOLD,

            window_size=
                STAGE9_WINDOW_SIZE,

            min_episodes=
                STAGE9_MIN_EPISODES,
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
        "Loading Stage 8A checkpoint: "
        f"{STAGE9_START_CHECKPOINT}"
    )

    ppo.load(
        STAGE9_START_CHECKPOINT,
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

    help_rate_window = deque(
        maxlen=100
    )

    global_step = 0

    for episode in range(
        1,
        STAGE9_MAX_EPISODES + 1,
    ):

        scenario = (
            stage9_sampler.sample()
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

        episode_help_count = 0

        episode_action_count = 0

        while env.agents:

            current_agents = (
                env.agents[:]
            )

            flat_observations = {}

            transition_data = {}

            actions = {}

            # ========================================
            # Policy actions
            #
            # Stage 9부터는 comm_action을 실제로
            # 환경에 전달한다 (이전 stage들과 다른
            # 부분은 여기뿐이다).
            # ========================================

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
                    comm_action,
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
                        comm_action,
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
                    comm_action,
                    log_prob,
                    value,
                )

                episode_action_count += 1

                if comm_action == HELP:

                    episode_help_count += 1

            # ========================================
            # Environment step
            # ========================================

            (
                next_observations,
                rewards,
                terminations,
                truncations,
                infos,
            ) = env.step(
                actions
            )

            # ========================================
            # Store transitions
            # ========================================

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
                    comm_action,
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

                    comm_action=
                        comm_action,

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

            info = (
                infos[
                    first_agent
                ]
            )

            episode_return += (
                rewards[first_agent]
            )

            episode_length += 1

            episode_success = (
                info["success"]
            )

            observations = (
                next_observations
            )

            # ========================================
            # PPO update
            # ========================================

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
                    metrics["entropy"],
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

        help_rate = (
            episode_help_count
            / max(
                episode_action_count,
                1,
            )
        )

        help_rate_window.append(
            help_rate
        )

        stage9_curriculum.record(
            episode_success
        )

        avg_length = np.mean(
            length_window
        )

        avg_return = np.mean(
            return_window
        )

        avg_help_rate = np.mean(
            help_rate_window
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
            int(episode_success),
            episode,
        )

        writer.add_scalar(
            "stage9/substage",
            stage9_sampler.level,
            episode,
        )

        writer.add_scalar(
            "stage9/success_rate",
            stage9_curriculum
            .success_rate,
            episode,
        )

        writer.add_scalar(
            "stage9/help_rate",
            help_rate,
            episode,
        )

        # ==================================================
        # Console
        # ==================================================

        if episode % 10 == 0:

            print(
                f"[Episode "
                f"{episode:5d}] "
                f"{stage9_sampler.name:<24} "
                f"success100="
                f"{stage9_curriculum.success_rate:.3f} "
                f"avg_len="
                f"{avg_length:.1f} "
                f"return="
                f"{avg_return:.3f} "
                f"help_rate="
                f"{avg_help_rate:.3f}"
            )

        # ==================================================
        # Periodic checkpoint
        # ==================================================

        if (
            episode % SAVE_INTERVAL
            == 0
        ):

            ppo.save(
                "checkpoints/"
                f"stage9_episode_"
                f"{episode}.pt"
            )

        # ==================================================
        # Stage 9 substage transition
        # ==================================================

        if stage9_curriculum.mastered():

            current_name = (
                stage9_sampler.name
            )

            print()
            print("=" * 70)

            print(
                f"{current_name} mastered! "
                f"success_rate="
                f"{stage9_curriculum.success_rate:.3f} "
                f"help_rate="
                f"{avg_help_rate:.3f}"
            )

            print("=" * 70)
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
                stage9_sampler.advance()
            )

            buffer.reset()

            length_window.clear()

            return_window.clear()

            help_rate_window.clear()

            stage9_curriculum.reset_level()

            if advanced:

                print(
                    f"Moving to "
                    f"{stage9_sampler.name}"
                )

            else:

                ppo.save(
                    "checkpoints/"
                    "stage_9_mastered.pt"
                )

                print(
                    "Stage 9 curriculum "
                    "completed! (pipeline "
                    "validation only - "
                    "see docs/"
                    "curriculum_plan.md)"
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
        "stage9_final.pt"
    )

    writer.close()

    env.close()

    print(
        "Stage 9 training complete."
    )


if __name__ == "__main__":
    main()
