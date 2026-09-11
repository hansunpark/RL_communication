import argparse
import os
import shutil

from evaluation.evaluator import (
    PolicyEvaluator,
)

from evaluation.benchmark_scenarios import (
    IID_SCENARIOS,
    OBSTACLE_SCENARIOS,
    TARGET_GOAL_SCENARIOS,
    COMBINED_SCENARIOS,
    STRESS_SCENARIOS,
)


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
        default="stochastic",
    )

    parser.add_argument(
        "--groups",
        nargs="+",
        choices=[
            "iid",
            "obstacle_ood",
            "target_goal_ood",
            "combined_ood",
            "stress",
            "all",
        ],
        default=["all"],
    )

    parser.add_argument(
        "--iid-episodes",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--episodes-per-scenario",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--no-trajectories",
        action="store_true",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # ========================================================
    # Groups
    # ========================================================

    if "all" in args.groups:

        groups = [
            "iid",
            "obstacle_ood",
            "target_goal_ood",
            "combined_ood",
            "stress",
        ]

    else:

        groups = list(
            args.groups
        )

    # ========================================================
    # Result directory
    # ========================================================

    output_root = os.path.join(
        "evaluation_results",
        args.mode,
    )

    trajectory_dir = os.path.join(
        output_root,
        "trajectories",
    )

    os.makedirs(
        output_root,
        exist_ok=True,
    )

    if not args.no_trajectories:

        os.makedirs(
            trajectory_dir,
            exist_ok=True,
        )

    # ========================================================
    # Freeze checkpoint
    #
    # 현재 실행에서 사용한 모델을 결과 폴더에 같이 복사.
    # 매번 덮어써서 stale checkpoint 문제 방지.
    # ========================================================

    frozen_checkpoint = os.path.join(
        output_root,
        "baseline_checkpoint.pt",
    )

    shutil.copy2(
        args.checkpoint,
        frozen_checkpoint,
    )

    print(
        f"Checkpoint copied:"
    )

    print(
        frozen_checkpoint
    )

    # ========================================================
    # Evaluator
    # ========================================================

    evaluator = PolicyEvaluator(
        checkpoint=
            frozen_checkpoint,

        device=
            args.device,

        action_mode=
            args.mode,
    )

    grouped_results = {}

    seed_base = 100000

    # ========================================================
    # IID
    # ========================================================

    if "iid" in groups:

        print()
        print(
            "#" * 70
        )

        print(
            "# IID STAGE 6"
        )

        print(
            "#" * 70
        )

        results = (
            evaluator.evaluate_scenarios(
                scenarios=
                    IID_SCENARIOS,

                episodes_per_scenario=
                    args.iid_episodes,

                seed_start=
                    seed_base,

                trajectory_dir=(
                    None
                    if args.no_trajectories
                    else trajectory_dir
                ),
            )
        )

        grouped_results[
            "iid"
        ] = results

        seed_base += (
            args.iid_episodes
            * len(
                IID_SCENARIOS
            )
            + 1000
        )

    # ========================================================
    # Obstacle OOD
    # ========================================================

    if "obstacle_ood" in groups:

        print()
        print(
            "#" * 70
        )

        print(
            "# OBSTACLE OOD"
        )

        print(
            "#" * 70
        )

        results = (
            evaluator.evaluate_scenarios(
                scenarios=
                    OBSTACLE_SCENARIOS,

                episodes_per_scenario=
                    args.
                    episodes_per_scenario,

                seed_start=
                    seed_base,

                trajectory_dir=(
                    None
                    if args.no_trajectories
                    else trajectory_dir
                ),
            )
        )

        grouped_results[
            "obstacle_ood"
        ] = results

        seed_base += (
            args.episodes_per_scenario
            *
            len(
                OBSTACLE_SCENARIOS
            )
            + 1000
        )

    # ========================================================
    # Target / Goal OOD
    # ========================================================

    if "target_goal_ood" in groups:

        print()
        print(
            "#" * 70
        )

        print(
            "# TARGET / GOAL OOD"
        )

        print(
            "#" * 70
        )

        results = (
            evaluator.evaluate_scenarios(
                scenarios=
                    TARGET_GOAL_SCENARIOS,

                episodes_per_scenario=
                    args.
                    episodes_per_scenario,

                seed_start=
                    seed_base,

                trajectory_dir=(
                    None
                    if args.no_trajectories
                    else trajectory_dir
                ),
            )
        )

        grouped_results[
            "target_goal_ood"
        ] = results

        seed_base += (
            args.episodes_per_scenario
            *
            len(
                TARGET_GOAL_SCENARIOS
            )
            + 1000
        )

    # ========================================================
    # Combined OOD
    # ========================================================

    if "combined_ood" in groups:

        print()
        print(
            "#" * 70
        )

        print(
            "# COMBINED OOD"
        )

        print(
            "#" * 70
        )

        results = (
            evaluator.evaluate_scenarios(
                scenarios=
                    COMBINED_SCENARIOS,

                episodes_per_scenario=
                    args.
                    episodes_per_scenario,

                seed_start=
                    seed_base,

                trajectory_dir=(
                    None
                    if args.no_trajectories
                    else trajectory_dir
                ),
            )
        )

        grouped_results[
            "combined_ood"
        ] = results

        seed_base += (
            args.episodes_per_scenario
            *
            len(
                COMBINED_SCENARIOS
            )
            + 1000
        )

    # ========================================================
    # Stress
    # ========================================================

    if "stress" in groups:

        print()
        print(
            "#" * 70
        )

        print(
            "# STRESS"
        )

        print(
            "#" * 70
        )

        results = (
            evaluator.evaluate_scenarios(
                scenarios=
                    STRESS_SCENARIOS,

                episodes_per_scenario=
                    args.
                    episodes_per_scenario,

                seed_start=
                    seed_base,

                trajectory_dir=(
                    None
                    if args.no_trajectories
                    else trajectory_dir
                ),
            )
        )

        grouped_results[
            "stress"
        ] = results

    # ========================================================
    # Merge results
    # ========================================================

    all_results = []

    for results in (
        grouped_results.values()
    ):
        all_results.extend(
            results
        )

    # ========================================================
    # CSV
    # ========================================================

    csv_path = os.path.join(
        output_root,
        "baseline_episodes.csv",
    )

    evaluator.save_csv(
        all_results,
        csv_path,
    )

    # ========================================================
    # JSON summary
    # ========================================================

    summary = evaluator.build_summary(
        grouped_results
    )

    summary_path = os.path.join(
        output_root,
        "baseline_summary.json",
    )

    evaluator.save_json(
        summary,
        summary_path,
    )

    # ========================================================
    # Finish
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        f"Finished"
    )

    print(
        f"Mode    : {args.mode}"
    )

    print(
        f"CSV     : {csv_path}"
    )

    print(
        f"Summary : {summary_path}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()