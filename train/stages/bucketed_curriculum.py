from collections import deque


# ============================================================
# Bucketed Curriculum Manager
#
# Stage 7의 LeftLaneFocusCurriculumManager
# (train/stages/stage_7/curriculum.py)가 lane별로 했던 것을
# 임의의 버킷 키에 대해 할 수 있도록 일반화한 것이다.
#
# 왜 필요한가:
# 평균 성공률만 보면 특정 조건(버킷) 하나가 완전히 실패해도
# 다른 조건들이 잘 되면 평균이 기준을 넘어서 mastered로
# 잘못 판정될 수 있다 (Stage 7 lane_left/x=2에서 실제로
# 발생했던 문제). 그래서 모든 버킷이 "개별적으로" 기준을
# 넘겨야만 mastered로 판정한다.
#
# 버킷은 record() 호출 시점에 처음 보는 키라면 자동으로
# 추가된다. 사전에 전체 버킷 목록을 알 필요가 없다 (obstacle
# 개수, target 크기 조합처럼 실행 중에만 정해지는 버킷도
# 지원하기 위함).
# ============================================================


class BucketedCurriculumManager:

    def __init__(
        self,
        threshold=0.85,
        window_size=100,
        min_total_episodes=800,
        thresholds=None,
    ):
        self.default_threshold = (
            threshold
        )

        self.thresholds = (
            thresholds or {}
        )

        self.window_size = window_size

        self.min_total_episodes = (
            min_total_episodes
        )

        self.windows = {}

        self.total_episodes = 0

        self.completed = False

    # --------------------------------------------------------
    # Recording
    # --------------------------------------------------------

    def record(self, bucket, success):

        if bucket not in self.windows:

            self.windows[bucket] = deque(
                maxlen=self.window_size
            )

        self.windows[bucket].append(
            float(success)
        )

        self.total_episodes += 1

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    def threshold_for(self, bucket):

        return self.thresholds.get(
            bucket,
            self.default_threshold,
        )

    def success_rate(self, bucket):

        window = self.windows.get(
            bucket
        )

        if not window:
            return 0.0

        return (
            sum(window)
            / len(window)
        )

    def bucket_ready(self, bucket):

        window = self.windows.get(
            bucket
        )

        if (
            window is None
            or
            len(window)
            < self.window_size
        ):
            return False

        return (
            self.success_rate(bucket)
            >=
            self.threshold_for(bucket)
        )

    def mastered(self):

        if (
            self.total_episodes
            < self.min_total_episodes
        ):
            return False

        if not self.windows:
            return False

        return all(
            self.bucket_ready(bucket)
            for bucket in self.windows
        )

    def status(self):

        return {
            bucket: {
                "n": len(window),

                "success_rate":
                    self.success_rate(
                        bucket
                    ),

                "ready":
                    self.bucket_ready(
                        bucket
                    ),
            }

            for bucket, window
            in self.windows.items()
        }

    # --------------------------------------------------------
    # Reset (다음 substage로 넘어갈 때)
    # --------------------------------------------------------

    def reset(self):

        self.windows.clear()

        self.total_episodes = 0
