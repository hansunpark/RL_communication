import argparse
import os
import shutil

from collections import defaultdict

from .benchmark_scenarios import (
    IID_SCENARIOS,
    OBSTACLE_SCENARIOS,
    TARGET_GOAL_SCENARIOS,
    COMBINED_SCENARIOS,
    STRESS_SCENARIOS,
)

from .evaluator import (
    PolicyEvaluator,
    save_csv,
    save_json,
)


def print_summary(
    name,
    summary,
):
    print()
    print(
        "-" * 65
    )

    print(name)

    print(
        "-" * 65
    )

    print(
        f"Episodes: "
        f"{summary['episodes']}"
    )

    print(
        f"Success Rate: "
        f"{summary['success_rate'] * 100:.2f}%"
    )

    print(
        f"Mean Episode Length: "
        f"{summary['mean_episode_length']:.2f}"
    )

    success_length = (
        summary[
            "mean_success_length"
        ]
    )

    if success_length is None:

        print(
            "Mean Success Length: "
            "N/A"
        )

    else:

        print(
            f"Mean Success Length: "
            f"{success_length:.2f}"
        )

    print(
        f"Mean Return: "
        f"{summary['mean_return']:.3f}"
    )

    print(
        f"Mean Target Moves: "
        f"{summary['mean_target_moves']:.2f}"
    )

    print(
        f"Mean Obstacle Moves: "
        f"{summary['mean_obstacle_moves']:.2f}"
    )


def run_group(
    evaluator,
    group_name,
    scenarios,
    episodes_per_scenario,
    seed_base,
    trajectory_dir,
    record_examples,
):
    group_rows = []

    scenario_summaries = {}

    for index, scenario in enumerate(
        scenarios
    ):
        print()
        print(
            f"[{group_name}] "
            f"{scenario.name}"
        )

        seed_start = (
            seed_base
            +
            index
            * 100000
        )

        results = (
            evaluator.evaluate_scenario(
                scenario=scenario,
                episodes=
                    episodes_per_scenario,
                seed_start=
                    seed_start,
                record_examples=
                    record_examples,
                trajectory_dir=
                    trajectory_dir,
            )
        )

        summary = (
            evaluator.summarize(
                results
            )
        )

        scenario_summaries[
            scenario.name
        ] = summary

        group_rows.extend(
            results
        )

        print(
            f"  success="
            f"{summary['success_rate'] * 100:.1f}% "
            f"mean_len="
            f"{summary['mean_episode_length']:.1f}"
        )

    group_summary = (
        evaluator.summarize(
            group_rows
        )
    )

    return (
        group_rows,
        group_summary,
        scenario_summaries,
    )


def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--checkpoint",

        type=str,

        default=(
            "checkpoints/"
            "stage_6_mastered.pt"
        ),
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
        "--output",

        type=str,

        default=(
            "evaluation_results"
        ),
    )

    parser.add_argument(
        "--no-trajectories",

        action="store_true",
    )

    args = parser.parse_args()

    os.makedirs(
        args.output,
        exist_ok=True,
    )

    trajectory_dir = (
        os.path.join(
            args.output,
            "trajectories",
        )
    )

    os.makedirs(
        trajectory_dir,
        exist_ok=True,
    )

    # ======================================================
    # Freeze baseline checkpoint
    # ======================================================

    baseline_checkpoint = (
        os.path.join(
            "checkpoints",
            "baseline_no_comm_stage6.pt",
        )
    )

    if (
        not os.path.exists(
            baseline_checkpoint
        )
    ):
        shutil.copy2(
            args.checkpoint,
            baseline_checkpoint,
        )

        print(
            f"Baseline checkpoint copied:\n"
            f"{baseline_checkpoint}"
        )

    else:

        print(
            f"Using existing baseline:\n"
            f"{baseline_checkpoint}"
        )

    evaluator = (
        PolicyEvaluator(
            checkpoint=
                baseline_checkpoint
        )
    )

    all_rows = []

    complete_summary = {}

    record_examples = (
        not args.no_trajectories
    )

    print()
    print(
        "=" * 70
    )

    print(
        "NO-COMMUNICATION BASELINE BENCHMARK"
    )

    print(
        "=" * 70
    )

    # ======================================================
    # A. IID
    # ======================================================

    iid_results = (
        evaluator.evaluate_scenario(
            scenario=
                IID_SCENARIOS[0],

            episodes=
                args.iid_episodes,

            seed_start=
                100000,

            record_examples=
                record_examples,

            trajectory_dir=
                trajectory_dir,
        )
    )

    iid_summary = (
        evaluator.summarize(
            iid_results
        )
    )

    all_rows.extend(
        iid_results
    )

    complete_summary[
        "iid"
    ] = {
        "overall":
            iid_summary,

        "scenarios": {
            "iid_stage6":
                iid_summary
        },
    }

    print_summary(
        "A. IID Stage 6",
        iid_summary,
    )

    # ======================================================
    # B-E
    # ======================================================

    groups = [
        (
            "obstacle_ood",
            "B. Obstacle Generalization",
            OBSTACLE_SCENARIOS,
            200000,
        ),

        (
            "target_goal_ood",
            "C. Target / Goal Generalization",
            TARGET_GOAL_SCENARIOS,
            300000,
        ),

        (
            "combined_ood",
            "D. Combined Generalization",
            COMBINED_SCENARIOS,
            400000,
        ),

        (
            "stress",
            "E. Stress Test",
            STRESS_SCENARIOS,
            500000,
        ),
    ]

    for (
        key,
        display_name,
        scenarios,
        seed_base,
    ) in groups:

        (
            rows,
            group_summary,
            scenario_summaries,
        ) = run_group(
            evaluator=evaluator,
            group_name=
                display_name,
            scenarios=scenarios,
            episodes_per_scenario=
                args.episodes_per_scenario,
            seed_base=
                seed_base,
            trajectory_dir=
                trajectory_dir,
            record_examples=
                record_examples,
        )

        all_rows.extend(
            rows
        )

        complete_summary[
            key
        ] = {
            "overall":
                group_summary,

            "scenarios":
                scenario_summaries,
        }

        print_summary(
            display_name,
            group_summary,
        )

    # ======================================================
    # Save
    # ======================================================

    csv_path = (
        os.path.join(
            args.output,
            "baseline_episodes.csv",
        )
    )

    json_path = (
        os.path.join(
            args.output,
            "baseline_summary.json",
        )
    )

    save_csv(
        all_rows,
        csv_path,
    )

    save_json(
        complete_summary,
        json_path,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "BENCHMARK COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Episode CSV:\n"
        f"{csv_path}"
    )

    print(
        f"\nSummary JSON:\n"
        f"{json_path}"
    )

    if record_examples:

        print(
            f"\nTrajectory examples:\n"
            f"{trajectory_dir}"
        )


if __name__ == "__main__":
    main()