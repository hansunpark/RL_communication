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


class TrajectoryVisualizer:

    def __init__(
        self,
        trajectory_path,
    ):

        with open(
            trajectory_path,
            "r",
            encoding="utf-8",
        ) as f:

            self.data = json.load(
                f
            )

        self.width = (
            self.data["width"]
        )

        self.height = (
            self.data["height"]
        )

        self.frames = (
            self.data["frames"]
        )

        self.fig, self.ax = (
            plt.subplots(
                figsize=(8, 8)
            )
        )

        # 매우 중요:
        # animation object가 garbage collection
        # 되는 것을 막는다.
        self.animation = None

    # ========================================================
    # Draw one frame
    # ========================================================

    def draw_frame(
        self,
        frame_index,
    ):

        self.ax.clear()

        frame = (
            self.frames[
                frame_index
            ]
        )

        # ----------------------------------------------------
        # Axis
        # ----------------------------------------------------

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
            True,
            linewidth=0.5,
        )

        # ----------------------------------------------------
        # Goal
        # ----------------------------------------------------

        for x, y in frame.get(
            "goal",
            [],
        ):

            rect = Rectangle(
                (x, y),
                1,
                1,
                fill=False,
                linewidth=2,
            )

            self.ax.add_patch(
                rect
            )

        # ----------------------------------------------------
        # Walls
        # ----------------------------------------------------

        for x, y in frame.get(
            "walls",
            [],
        ):

            rect = Rectangle(
                (x, y),
                1,
                1,
            )

            self.ax.add_patch(
                rect
            )

        # ----------------------------------------------------
        # Objects
        # ----------------------------------------------------

        for obj in frame.get(
            "objects",
            [],
        ):

            rect = Rectangle(
                (
                    obj["x"],
                    obj["y"],
                ),
                obj["width"],
                obj["height"],
                fill=False,
                linewidth=3,
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
                f"O{obj['id']}"
            )

            self.ax.text(
                obj["x"]
                + obj["width"] / 2,

                obj["y"]
                + obj["height"] / 2,

                label,

                horizontalalignment=
                    "center",

                verticalalignment=
                    "center",
            )

        # ----------------------------------------------------
        # Agents
        # ----------------------------------------------------

        for (
            agent,
            position,
        ) in frame.get(
            "agents",
            {},
        ).items():

            x, y = position

            self.ax.scatter(
                x + 0.5,
                y + 0.5,
                s=200,
            )

            action_text = ""

            action_data = (
                frame.get(
                    "actions",
                    {},
                ).get(
                    agent
                )
            )

            if (
                action_data
                is not None
            ):

                if isinstance(
                    action_data,
                    list,
                ):
                    physical_action = (
                        action_data[0]
                    )

                else:
                    physical_action = (
                        action_data
                    )

                action_text = (
                    ACTION_NAMES.get(
                        physical_action,
                        str(
                            physical_action
                        ),
                    )
                )

            self.ax.text(
                x + 0.5,
                y + 0.25,
                agent.replace(
                    "agent_",
                    "A",
                ),

                horizontalalignment=
                    "center",
            )

            self.ax.text(
                x + 0.5,
                y + 0.78,
                action_text,

                horizontalalignment=
                    "center",

                fontsize=8,
            )

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        success = frame.get(
            "success",
            False,
        )

        target_distance = (
            frame.get(
                "target_distance"
            )
        )

        blocking = frame.get(
            "obstacle_blocking"
        )

        self.ax.set_title(
            f"{self.data['scenario']} | "
            f"{self.data['action_mode']}\n"
            f"step={frame_index} | "
            f"target_distance="
            f"{target_distance} | "
            f"blocking={blocking} | "
            f"success={success}"
        )

    # ========================================================
    # GUI
    # ========================================================

    def show(
        self,
        interval=250,
    ):

        self.animation = (
            FuncAnimation(
                self.fig,
                self.draw_frame,
                frames=
                    len(
                        self.frames
                    ),
                interval=
                    interval,
                repeat=False,
            )
        )

        plt.show()

    # ========================================================
    # GIF
    # ========================================================

    def save_gif(
        self,
        output_path,
        fps=5,
    ):

        self.animation = (
            FuncAnimation(
                self.fig,
                self.draw_frame,
                frames=
                    len(
                        self.frames
                    ),
                interval=
                    1000 / fps,
                repeat=False,
            )
        )

        writer = (
            PillowWriter(
                fps=fps
            )
        )

        self.animation.save(
            output_path,
            writer=writer,
        )

        print(
            f"GIF saved: "
            f"{output_path}"
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trajectory",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--gif",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    visualizer = (
        TrajectoryVisualizer(
            args.trajectory
        )
    )

    if args.gif:

        visualizer.save_gif(
            output_path=
                args.gif,

            fps=
                args.fps,
        )

    else:

        visualizer.show()


if __name__ == "__main__":
    main()