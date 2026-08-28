import argparse
import json

import matplotlib.pyplot as plt

from matplotlib.animation import (
    FuncAnimation,
    PillowWriter,
)

from matplotlib.patches import (
    Rectangle,
)


ACTION_NAMES = {
    0: "STAY",
    1: "UP",
    2: "RIGHT",
    3: "DOWN",
    4: "LEFT",
}


def load_trajectory(
    path,
):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


class TrajectoryVisualizer:

    def __init__(
        self,
        trajectory,
    ):
        self.data = trajectory

        self.width = (
            trajectory["width"]
        )

        self.height = (
            trajectory["height"]
        )

        self.frames = (
            trajectory["frames"]
        )

        self.fig, self.ax = (
            plt.subplots(
                figsize=(8, 8)
            )
        )
        self.animation=None

    def draw_frame(
        self,
        index,
    ):
        frame = (
            self.frames[index]
        )

        self.ax.clear()

        self.ax.set_xlim(
            0,
            self.width,
        )

        self.ax.set_ylim(
            self.height,
            0,
        )

        self.ax.set_aspect(
            "equal"
        )

        self.ax.set_xticks(
            range(
                self.width + 1
            )
        )

        self.ax.set_yticks(
            range(
                self.height + 1
            )
        )

        self.ax.grid(
            True
        )

        self.ax.set_xticklabels(
            []
        )

        self.ax.set_yticklabels(
            []
        )

        # ==============================================
        # Goal
        # ==============================================

        for x, y in (
            frame[
                "goal_cells"
            ]
        ):
            rect = Rectangle(
                (x, y),
                1,
                1,
                alpha=0.25,
            )

            self.ax.add_patch(
                rect
            )

            self.ax.text(
                x + 0.5,
                y + 0.5,
                "G",
                ha="center",
                va="center",
                fontsize=8,
            )

        # ==============================================
        # Walls
        # ==============================================

        for x, y in (
            frame["walls"]
        ):
            rect = Rectangle(
                (x, y),
                1,
                1,
                alpha=0.8,
            )

            self.ax.add_patch(
                rect
            )

        # ==============================================
        # Objects
        # ==============================================

        for obj in (
            frame["objects"]
        ):
            rect = Rectangle(
                (
                    obj["x"],
                    obj["y"],
                ),
                obj["width"],
                obj["height"],
                alpha=0.55,
            )

            self.ax.add_patch(
                rect
            )

            label = (
                "TARGET"
                if obj[
                    "is_target"
                ]
                else
                f"OBS {obj['id']}"
            )

            self.ax.text(
                obj["x"]
                +
                obj["width"]
                / 2,

                obj["y"]
                +
                obj["height"]
                / 2,

                label,

                ha="center",
                va="center",
                fontsize=9,
                weight="bold",
            )

        # ==============================================
        # Agents
        # ==============================================

        actions = (
            frame.get(
                "actions"
            )
        )

        for agent, (
            x,
            y,
        ) in (
            frame[
                "agents"
            ].items()
        ):
            circle = (
                plt.Circle(
                    (
                        x + 0.5,
                        y + 0.5,
                    ),
                    0.32,
                    fill=False,
                    linewidth=2,
                )
            )

            self.ax.add_patch(
                circle
            )

            label = agent.replace(
                "agent_",
                "A",
            )

            if actions:

                action_id = (
                    actions[
                        agent
                    ][0]
                )

                action_name = (
                    ACTION_NAMES[
                        action_id
                    ]
                )

                label += (
                    f"\n{action_name}"
                )

            self.ax.text(
                x + 0.5,
                y + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=8,
            )

        # ==============================================
        # Information
        # ==============================================

        title = (
            f"{self.data['scenario']} | "
            f"seed={self.data['seed']} | "
            f"step={frame['step']}"
        )

        if (
            "target_distance"
            in frame
        ):
            title += (
                f"\nreward="
                f"{frame['reward']:.2f}"
                f" | distance="
                f"{frame['target_distance']}"
                f" | blocking="
                f"{frame['obstacle_blocking']}"
            )

        self.ax.set_title(
            title
        )

    def show(self, interval=250):

        self.animation = FuncAnimation(
            self.fig,
            self.draw_frame,
            frames=len(self.frames),
            interval=interval,
            repeat=False,
        )

        plt.show()

        return self.animation
    
    def save_gif(
        self,
        output_path,
        fps=5,
    ):
        self.animation = FuncAnimation(
            self.fig,
            self.draw_frame,
            frames=len(self.frames),
            interval=1000 / fps,
            repeat=False,
        )

        writer = PillowWriter(
            fps=fps
        )

        self.animation.save(
            output_path,
            writer=writer,
        )

        print(
            f"GIF saved to {output_path}"
        )


def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--trajectory",
        required=True,
    )

    parser.add_argument(
        "--gif",
        default=None,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    trajectory = (
        load_trajectory(
            args.trajectory
        )
    )

    visualizer = (
        TrajectoryVisualizer(
            trajectory
        )
    )

    if args.gif:

        visualizer.save_gif(
            args.gif,
            fps=args.fps,
        )

    else:

        visualizer.show()


if __name__ == "__main__":
    main()