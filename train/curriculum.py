from collections import deque


class CurriculumManager:

    def __init__(
        self,
        success_threshold=0.85,
        window_size=100,
        min_episodes=200,
        stage4_baseline_patience=600,
    ):

        self.success_threshold = (
            success_threshold
        )

        self.window_size = (
            window_size
        )

        self.min_episodes = (
            min_episodes
        )

        self.stage4_baseline_patience = (
            stage4_baseline_patience
        )

        self.stage = 0

        self.spawn_level = 0

        self.stage4_phase = (
            "target_only"
        )

        self.success_window = deque(
            maxlen=window_size
        )

        self.episodes_in_unit = 0

        self.completed = False

    # ======================================================
    # Current environment configuration
    # ======================================================

    @property
    def reward_mode(self):

        if (
            self.stage == 4
            and
            self.stage4_phase
            == "obstacle_shaping"
        ):
            return (
                "obstacle_shaping"
            )

        if self.stage >= 5:

            return (
                "obstacle_shaping"
            )

        return (
            "target_only"
        )

    @property
    def success_rate(self):

        if (
            len(
                self.success_window
            )
            == 0
        ):
            return 0.0

        return (
            sum(
                self.success_window
            )
            /
            len(
                self.success_window
            )
        )

    @property
    def unit_name(self):

        if self.stage == 3:

            names = [
                "stage_3_radius3",
                "stage_3_radius5",
                "stage_3_full",
            ]

            return names[
                self.spawn_level
            ]

        if self.stage == 4:

            return (
                "stage_4_"
                +
                self.stage4_phase
            )

        return (
            f"stage_{self.stage}"
        )

    def config(self):

        return {
            "stage":
                self.stage,

            "spawn_level":
                self.spawn_level,

            "reward_mode":
                self.reward_mode,
        }

    # ======================================================
    # Episode update
    # ======================================================

    def record_episode(
        self,
        success,
    ):

        if self.completed:

            return None

        self.success_window.append(
            int(success)
        )

        self.episodes_in_unit += 1

        enough_window = (
            len(
                self.success_window
            )
            >= self.window_size
        )

        enough_episodes = (
            self.episodes_in_unit
            >= self.min_episodes
        )

        mastered = (
            enough_window
            and
            enough_episodes
            and
            self.success_rate
            >= self.success_threshold
        )

        # ==================================================
        # Mastery
        # ==================================================

        if mastered:

            old_name = (
                self.unit_name
            )

            # Stage 3 has three sub-levels
            if (
                self.stage == 3
                and
                self.spawn_level < 2
            ):

                self.spawn_level += 1

                self._reset_statistics()

                return {
                    "changed": True,

                    "type":
                        "difficulty_up",

                    "checkpoint":
                        old_name
                        + "_mastered.pt",

                    "message":
                        (
                            f"{old_name} mastered "
                            f"-> {self.unit_name}"
                        ),
                }

            # Final stage
            if self.stage == 6:

                self.completed = True

                return {
                    "changed": True,

                    "type":
                        "completed",

                    "checkpoint":
                        "stage_6_mastered.pt",

                    "message":
                        (
                            "Final Stage 6 "
                            "mastered."
                        ),
                }

            # Normal stage up
            self.stage += 1

            self.spawn_level = 0

            if self.stage == 4:

                self.stage4_phase = (
                    "target_only"
                )

            self._reset_statistics()

            return {
                "changed": True,

                "type":
                    "stage_up",

                "checkpoint":
                    old_name
                    + "_mastered.pt",

                "message":
                    (
                        f"{old_name} mastered "
                        f"-> {self.unit_name}"
                    ),
            }

        # ==================================================
        # Stage 4 baseline failure
        # ==================================================

        if (
            self.stage == 4
            and
            self.stage4_phase
            == "target_only"
            and
            self.episodes_in_unit
            >=
            self.stage4_baseline_patience
        ):

            old_rate = (
                self.success_rate
            )

            self.stage4_phase = (
                "obstacle_shaping"
            )

            self._reset_statistics()

            return {
                "changed": True,

                "type":
                    "enable_shaping",

                "checkpoint":
                    (
                        "stage_4_"
                        "target_only_"
                        "failed.pt"
                    ),

                "message":
                    (
                        "Stage 4 target-only "
                        "did not reach mastery. "
                        f"success100="
                        f"{old_rate:.3f}. "
                        "Obstacle shaping enabled."
                    ),
            }

        return None

    def _reset_statistics(self):

        self.success_window.clear()

        self.episodes_in_unit = 0