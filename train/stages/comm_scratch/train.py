import argparse

from ..comm_runner import (
    RunConfig,
    train,
)

from .curriculum import (
    CommScratchSampler,
)


# ============================================================
# comm_scratch 학습 (처음부터 통신 on, 새 커리큘럼)
#
# 실행:
#   python -m train.stages.comm_scratch.train
#   python -m train.stages.comm_scratch.train --resume
#
# 무작위 초기화에서 시작한다 (체크포인트 없음). 네트워크 구조와
# 관측/보상은 기존 모델과 동일하게 유지해서, 기존 no-comm 모델 및
# stage8_comm 트랙과 비교할 수 있게 한다.
#
# help_penalty는 레벨별로 curriculum.py의 LEVELS에서 정한다
# (정보 비대칭이 생기기 전까지 0).
# ============================================================

RUN_NAME = "comm_scratch"

ENV_KWARGS = dict(
    num_agents=4,
    vision_size=5,
    max_steps=200,
    curriculum_stage=0,
    spawn_level=0,
    # obstacle이 없는 레벨에서는 blocking score가 항상 0이라
    # target_only와 같다. obstacle 레벨에서 다시 바꿀 필요가
    # 없도록 처음부터 obstacle_shaping으로 둔다.
    reward_mode="obstacle_shaping",
    step_penalty=-0.01,
    distance_reward_coef=0.2,
    obstacle_reward_coef=0.1,
    success_reward=10.0,
    help_penalty=0.0,
)


def make_config():

    return RunConfig(
        run_name=RUN_NAME,
        start_checkpoint=None,
        # 3e-4 -> 1e-4 + target_kl 0.02: S3, S3a, S4에서 반복해서
        # 이미 하던 과제까지 잃으며 무너졌다 (S4 15,000 episode 후 S3b
        # 과제 85% -> 18%). 같은 기간 lr 1e-4인 stage8_comm은 한 번도
        # 무너지지 않았다. docs/curriculum_plan.md 참고.
        learning_rate=1e-4,
        target_kl=0.02,
        entropy_coef=0.01,
        comm_entropy_coef=0.01,
        comm_enabled=True,
        # 이미 통과한 레벨을 25% 섞어서, 실패 위주 구간에서
        # 기존 능력을 잃지 않게 한다 (comm_runner.RunConfig 참고).
        rehearsal_prob=0.25,
        env_kwargs=ENV_KWARGS,
        # 병목이 Python env라서 GPU가 오히려 느리다
        # (16 env 기준 CPU 3,827 vs GPU 2,572 env-step/s).
        device="cpu",
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    # 이미 통과한 레벨의 _mastered.pt에서 이어갈 때 쓴다.
    parser.add_argument(
        "--start-level",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--start-checkpoint",
        default=None,
    )

    # 지금 레벨만 마저 학습하고 멈출 때 쓴다.
    parser.add_argument(
        "--stop-after-current-level",
        action="store_true",
    )

    # GRU 메모리 정책(RecurrentPPO)으로 학습한다. 실행 이름에
    # "_gru"가 붙어서 체크포인트/로그/TensorBoard가 MLP 실행과
    # 섞이지 않는다. MLP 체크포인트를 --start-checkpoint로 주면
    # 처음에는 MLP와 똑같이 행동하는 상태에서 시작한다.
    parser.add_argument(
        "--recurrent",
        action="store_true",
    )

    args = parser.parse_args()

    config = make_config()

    if args.start_level is not None:
        config.start_level = (
            args.start_level
        )

    if args.start_checkpoint:
        config.start_checkpoint = (
            args.start_checkpoint
        )

    config.stop_after_current_level = (
        args.stop_after_current_level
    )

    if args.recurrent:

        config.recurrent = True

        config.run_name += "_gru"

    train(
        config,
        CommScratchSampler(),
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
