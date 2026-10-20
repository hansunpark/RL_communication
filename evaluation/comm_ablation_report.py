import argparse
import json
import os

import torch

from gymnasium.spaces import flatdim

from env import CooperativeTransportEnv

from train.ppo import PPO
from train.recurrent_ppo import RecurrentPPO

from train.stages.comm_runner import (
    evaluate_level,
)


# ============================================================
# 통신 on/off 비교 리포트 (성공률, RC, CPR)
#
# 레벨을 통과한 체크포인트들을 그 레벨에서 다시 평가한다. 학습 중
# ablation과 같은 evaluate_level을 쓰므로 같은 시나리오 목록으로
# 통신 on/off를 비교한다.
#
# 사용:
#   python -m evaluation.comm_ablation_report --track B
#   python -m evaluation.comm_ablation_report --track B --run comm_scratch_gru --recurrent
#
# 결과: evaluation_results/<run>/comm_report.json, comm_report.md
# ============================================================


def _track(name):

    if name == "A":
        from train.stages.stage_8_comm.curriculum import (
            Stage8CommSampler,
        )

        from train.stages.stage_8_comm.train import (
            ENV_KWARGS,
            HELP_PENALTY,
        )

        return (
            "stage8_comm",
            lambda: Stage8CommSampler(
                help_penalty=HELP_PENALTY
            ),
            ENV_KWARGS,
        )

    from train.stages.comm_scratch.curriculum import (
        CommScratchSampler,
    )

    from train.stages.comm_scratch.train import (
        ENV_KWARGS,
    )

    return (
        "comm_scratch",
        CommScratchSampler,
        ENV_KWARGS,
    )


def _fmt(value):

    return "-" if value is None else f"{value:.2f}"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--track",
        choices=["A", "B"],
        required=True,
    )

    parser.add_argument("--run")

    parser.add_argument(
        "--recurrent",
        action="store_true",
    )

    parser.add_argument(
        "--levels",
        nargs="*",
        help="레벨 이름 (생략하면 _mastered.pt가 있는 레벨 전부)",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    args = parser.parse_args()

    default_run, make_sampler, env_kwargs = (
        _track(args.track)
    )

    run = args.run or default_run

    torch.set_num_threads(4)

    probe = CooperativeTransportEnv(
        **env_kwargs
    )

    obs_dim = flatdim(
        probe.observation_space(
            probe.possible_agents[0]
        )
    )

    sampler = make_sampler()

    levels = args.levels or [
        name
        for name in sampler.LEVEL_NAMES
        if os.path.exists(
            f"checkpoints/{run}_{name}_mastered.pt"
        )
    ]

    rows = []

    for name in levels:

        ppo = (
            RecurrentPPO(
                obs_dim=obs_dim,
                device=args.device,
            )
            if args.recurrent
            else PPO(
                obs_dim=obs_dim,
                device=args.device,
            )
        )

        ppo.load(
            f"checkpoints/{run}_{name}_mastered.pt",
            load_optimizer=False,
        )

        sampler.level = sampler.LEVEL_NAMES.index(
            name
        )

        result = {"level": name}

        for side, comm in (
            ("comm_on", True),
            ("comm_off", False),
        ):
            result[side] = evaluate_level(
                ppo=ppo,
                sampler=sampler,
                env_kwargs=env_kwargs,
                episodes=args.episodes,
                comm_enabled=comm,
                seed=42 * 7919 + 17,
            )

        rows.append(result)

        on, off = (
            result["comm_on"],
            result["comm_off"],
        )

        print(
            f"{name:48s} "
            f"success {on['success_rate']:.2f}/{off['success_rate']:.2f} "
            f"RC {_fmt(on['rc'])}/{_fmt(off['rc'])} "
            f"CPR {_fmt(on['cpr'])}/{_fmt(off['cpr'])} "
            f"rally {_fmt(on['rally_steps'])}/{_fmt(off['rally_steps'])}",
            flush=True,
        )

    out_dir = os.path.join(
        "evaluation_results",
        run,
    )

    os.makedirs(out_dir, exist_ok=True)

    with open(
        os.path.join(out_dir, "comm_report.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(rows, file, indent=2)

    lines = [
        "| 레벨 | 성공률 on | off | RC on | off | CPR on | off | 모이기 step on | off |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for row in rows:

        on, off = row["comm_on"], row["comm_off"]

        lines.append(
            f"| {row['level']} "
            f"| {on['success_rate']:.2f} | {off['success_rate']:.2f} "
            f"| {_fmt(on['rc'])} | {_fmt(off['rc'])} "
            f"| {_fmt(on['cpr'])} | {_fmt(off['cpr'])} "
            f"| {_fmt(on['rally_steps'])} | {_fmt(off['rally_steps'])} |"
        )

    with open(
        os.path.join(out_dir, "comm_report.md"),
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
