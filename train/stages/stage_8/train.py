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

from ..bucketed_curriculum import (
    BucketedCurriculumManager,
)

from .curriculum import (
    Stage8ScenarioSampler,
)


# ==========================================================
# Configuration
#
# Stage 8은 Stage 7 산출물(stage_7_mastered.pt)에서 이어서,
# 통신은 계속 NO_MESSAGE로 고정한 채 공간 일반화만 검증한다
# (target 크기/위치, obstacle 개수/크기/위치 랜덤화).
#
# docs/curriculum_plan.md의 Phase 1 참고.
# ==========================================================

SEED = 42

STAGE8_MAX_EPISODES = 40000

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

# 버킷(target 크기 x obstacle 개수)별 독립 판정 기준.
# Stage 7에서 평균 성공률이 lane 하나의 완전 실패를 가렸던
# 실패를 반복하지 않기 위해 BucketedCurriculumManager를 쓴다.
STAGE8_SUCCESS_THRESHOLD = 0.85

STAGE8_WINDOW_SIZE = 100

# 버킷 수가 substage마다 다르므로(8A/8B는 2개, 8C/8D는
# 최대 10개) 넉넉하게 잡는다. 버킷마다 window가 다 차야
# mastered 판정이 가능하므로, 버킷 수가 많을수록 실제로는
# 이보다 훨씬 많은 episode가 걸릴 수 있다.
STAGE8_MIN_TOTAL_EPISODES = 1500

STAGE8_START_CHECKPOINT = (
    "checkpoints/stage_7_mastered.pt"
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
        "stage8"
    )

    # ------------------------------------------------------
    # Environment
    #
    # curriculum_stage/spawn_level은 생성자 요건을 채우기
    # 위한 값일 뿐이다. 모든 episode는 아래에서 항상
    # options={"scenario": ...}로 reset되므로 실제로는
    # Stage8ScenarioSampler가 만든 커스텀 시나리오만 쓰인다.
    # ------------------------------------------------------

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

    stage8_sampler = (
        Stage8ScenarioSampler(
            map_width=env.width,
            map_height=env.height,
        )
    )

    stage8_curriculum = (
        BucketedCurriculumManager(
            threshold=
                STAGE8_SUCCESS_THRESHOLD,

            window_size=
                STAGE8_WINDOW_SIZE,

            min_total_episodes=
                STAGE8_MIN_TOTAL_EPISODES,
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
        "Loading Stage 7 checkpoint: "
        f"{STAGE8_START_CHECKPOINT}"
    )

    ppo.load(
        STAGE8_START_CHECKPOINT,
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
        STAGE8_MAX_EPISODES + 1,
    ):

        scenario = (
            stage8_sampler.sample()
        )

        bucket = (
            stage8_sampler
            .bucket_for(scenario)
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

            # ========================================
            # Policy actions
            #
            # 통신 액션은 Stage 8 전체 기간 동안
            # 항상 NO_MESSAGE로 고정한다 (Phase 1은
            # 통신 없이 공간 일반화만 검증).
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

        stage8_curriculum.record(
            bucket,
            episode_success,
        )

        avg_length = np.mean(
            length_window
        )

        avg_return = np.mean(
            return_window
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
            "stage8/substage",
            stage8_sampler.level,
            episode,
        )

        writer.add_scalar(
            f"stage8/success_rate/"
            f"{bucket}",

            stage8_curriculum
            .success_rate(bucket),

            episode,
        )

        # ==================================================
        # Console
        # ==================================================

        if episode % 10 == 0:

            print(
                f"[Episode "
                f"{episode:5d}] "
                f"{stage8_sampler.name:<20} "
                f"bucket="
                f"{bucket:<14} "
                f"bucket_success="
                f"{stage8_curriculum.success_rate(bucket):.3f} "
                f"total_episodes="
                f"{stage8_curriculum.total_episodes:5d} "
                f"avg_len="
                f"{avg_length:.1f} "
                f"return="
                f"{avg_return:.3f}"
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
                f"stage8_episode_"
                f"{episode}.pt"
            )

        # ==================================================
        # Stage 8 substage transition
        # ==================================================

        if stage8_curriculum.mastered():

            current_name = (
                stage8_sampler.name
            )

            print()
            print("=" * 70)

            print(
                f"{current_name} mastered!"
            )

            for (
                bucket_name,
                bucket_status,
            ) in (
                stage8_curriculum
                .status()
                .items()
            ):

                print(
                    f"  {bucket_name:<16} "
                    f"n={bucket_status['n']:4d} "
                    f"success_rate="
                    f"{bucket_status['success_rate']:.3f}"
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
                stage8_sampler.advance()
            )

            # 이전 substage의 rollout을
            # 다음 substage와 섞지 않는다.
            buffer.reset()

            length_window.clear()

            return_window.clear()

            stage8_curriculum.reset()

            if advanced:

                print(
                    f"Moving to "
                    f"{stage8_sampler.name}"
                )

            else:

                ppo.save(
                    "checkpoints/"
                    "stage_8_mastered.pt"
                )

                stage8_curriculum.completed = (
                    True
                )

                print(
                    "Stage 8 curriculum "
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
        "stage8_final.pt"
    )

    writer.close()

    env.close()

    print(
        "Stage 8 training complete."
    )


if __name__ == "__main__":
    main()
