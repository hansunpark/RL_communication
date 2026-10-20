import argparse

from ..comm_runner import (
    RunConfig,
    train,
)

from .curriculum import (
    Stage8CommSampler,
)


# ============================================================
# Stage 8-comm 학습 (8B1부터 통신 on)
#
# 실행:
#   python -m train.stages.stage_8_comm.train
#   python -m train.stages.stage_8_comm.train --resume
#
# Stage 8(no-comm) 대비 바꾼 하이퍼파라미터와 이유:
# - entropy_coef 0.03 -> 0.005 (이동 head):
#   이전 8B1 시도에서 이동 entropy가 0.7 -> 1.4까지 치솟으며
#   warm-start 정책이 거의 무작위가 됐고 성공률이 40% -> 4%로
#   무너졌다 (runs/stage8 TensorBoard 기록). warm-start 정책의
#   기존 능력을 지키기 위해 낮춘다.
# - comm_entropy_coef 0.01 (통신 head 별도):
#   comm head는 새로 초기화되므로, 쓸모를 배우기 전에 HELP가
#   0%/100%로 굳지 않도록 따로 entropy를 준다.
# - learning_rate 1e-4 + critic_warmup_updates 30:
#   1차 실행(lr 2e-4, warmup 없음)에서 entropy 폭주는 없었지만
#   8B1 성공률이 24% -> 5%로 떨어졌다. 같은 체크포인트를 통신
#   on/off로 평가해도 둘 다 떨어져 있어서(0.077 / 0.040), 통신
#   입력 때문이 아니라 fine-tune 자체가 정책을 망가뜨린 것이다.
#   8A critic이 "거의 항상 성공"에 맞춰져 있어서 초반 advantage가
#   체계적으로 음수가 되는 것이 가장 유력한 원인이라, 처음 30번의
#   update는 critic head만 다시 맞춘다.
# ============================================================

RUN_NAME = "stage8_comm"

START_CHECKPOINT = (
    "checkpoints/"
    "stage_8A_size_mastered.pt"
)

HELP_PENALTY = 0.002

ENV_KWARGS = dict(
    num_agents=4,
    vision_size=5,
    max_steps=200,
    curriculum_stage=6,
    spawn_level=2,
    reward_mode="obstacle_shaping",
    step_penalty=-0.01,
    distance_reward_coef=0.2,
    obstacle_reward_coef=0.1,
    success_reward=10.0,
    help_penalty=HELP_PENALTY,
)


def make_config():

    return RunConfig(
        run_name=RUN_NAME,
        start_checkpoint=
            START_CHECKPOINT,
        learning_rate=1e-4,
        # comm_scratch에서 붕괴를 막은 KL 조기 종료를 여기도 건다
        # (8C0a 붕괴 이후 추가).
        target_kl=0.02,
        critic_warmup_updates=30,
        # 0.005 -> 0.01: 8B0에서 "옆으로 밀기"를 새로 발견해야
        # 하므로 탐색을 조금 더 준다 (폭주했던 0.03보다는 훨씬 낮게).
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
        Stage8CommSampler(
            help_penalty=HELP_PENALTY,
        ),
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
