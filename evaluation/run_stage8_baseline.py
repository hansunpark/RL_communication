import argparse
import os

from evaluation.evaluator import PolicyEvaluator
from evaluation.benchmark_scenarios import BenchmarkScenario

from train.stages.stage_8.curriculum import Stage8ScenarioSampler


# ============================================================
# Stage 8 축(target 크기/위치, obstacle 개수) 전체에 대해
# 체크포인트 하나를 평가해서 레벨별/버킷별 성공률을 남긴다.
#
# 나중에 통신(HELP)을 켜고 재학습했을 때, 통신이 실제로
# 도움이 됐는지 비교할 "통신 없는 상태의 베이스라인"으로 쓴다.
#
# 기존 evaluation/run_baseline_benchmark.py는 Stage 6/7의
# 고정 기하학 벤치마크(IID/OOD)용이라 Stage 8의 "매 episode
# 다른 랜덤 기하학"에는 맞지 않는다. 이 스크립트는
# Stage8ScenarioSampler로 각 레벨(8A~8D)마다 매 episode 새
# 시나리오를 뽑아서 evaluator.run_episode()에 넘긴다.
# ============================================================


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "stochastic",
            "deterministic",
        ],
        default="deterministic",
    )

    parser.add_argument(
        "--episodes-per-level",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=500000,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    evaluator = PolicyEvaluator(
        checkpoint=args.checkpoint,
        device=args.device,
        action_mode=args.mode,

        # Stage 8 체크포인트는 통신 head를 학습한 적이
        # 없다(무작위 초기값). 이 baseline은 "통신 도입 전"
        # 성능을 나중과 비교하기 위한 기준점이므로 HELP를
        # 강제로 꺼서 노이즈를 없앤다.
        disable_communication=True,
    )

    sampler = Stage8ScenarioSampler(
        map_width=evaluator.env.width,
        map_height=evaluator.env.height,
    )

    all_results = []

    seed = args.seed_start

    for level in range(
        len(
            Stage8ScenarioSampler
            .LEVEL_NAMES
        )
    ):

        sampler.level = level

        level_name = sampler.name

        level_results = []

        for _ in range(
            args.episodes_per_level
        ):

            scenario_dict = (
                sampler.sample()
            )

            bucket = (
                sampler.bucket_for(
                    scenario_dict
                )
            )

            benchmark_scenario = (
                BenchmarkScenario(
                    name=level_name,
                    config=scenario_dict,
                )
            )

            result = (
                evaluator.run_episode(
                    scenario=
                        benchmark_scenario,
                    seed=seed,
                )
            )

            result["level"] = level_name
            result["bucket"] = bucket

            all_results.append(result)
            level_results.append(result)

            seed += 1

        summary = PolicyEvaluator.aggregate(
            level_results
        )

        print(
            f"{level_name:<30} "
            f"success_rate="
            f"{summary['success_rate']:.3f} "
            f"n={summary['episodes']}"
        )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    checkpoint_name = (
        os.path.splitext(
            os.path.basename(
                args.checkpoint
            )
        )[0]
    )

    output_dir = (
        args.output_dir
        or os.path.join(
            "evaluation_results",
            "stage8_baseline",
            checkpoint_name,
            args.mode,
        )
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    PolicyEvaluator.save_csv(
        all_results,
        os.path.join(
            output_dir,
            "episodes.csv",
        ),
    )

    by_level = {}
    by_bucket = {}

    for result in all_results:

        by_level.setdefault(
            result["level"],
            [],
        ).append(result)

        by_bucket.setdefault(
            result["bucket"],
            [],
        ).append(result)

    summary = {
        "checkpoint": args.checkpoint,
        "mode": args.mode,

        "overall":
            PolicyEvaluator.aggregate(
                all_results
            ),

        "by_level": {
            level_name:
                PolicyEvaluator.aggregate(
                    results
                )

            for level_name, results
            in by_level.items()
        },

        "by_bucket": {
            bucket_name:
                PolicyEvaluator.aggregate(
                    results
                )

            for bucket_name, results
            in by_bucket.items()
        },
    }

    PolicyEvaluator.save_json(
        summary,
        os.path.join(
            output_dir,
            "summary.json",
        ),
    )

    print()
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
