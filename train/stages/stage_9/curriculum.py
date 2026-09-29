from .scenario import make_stage9_scenario


# ============================================================
# Stage 9 Level Sampler
#
# near_count가 클수록(= far 에이전트가 적을수록) 쉬운 문제다.
# Stage 8 8A -> 8B1 사고(한 번에 두 축을 도입해서 탐색이
# 막혔던 것)를 반복하지 않기 위해 쉬운 쪽(3near/1far)부터
# 시작해서 어려운 쪽(1near/3far)으로 진행한다.
#
#   9A: 3 near, 1 far  (가장 쉬움 - 거의 기존과 동일)
#   9B: 2 near, 2 far
#   9C: 1 near, 3 far  (가장 어려움 - 원래 계획서의 "9A")
# ============================================================


class Stage9LevelSampler:

    LEVELS = [
        {
            "name": "stage_9A_3near_1far",
            "near_count": 3,
        },
        {
            "name": "stage_9B_2near_2far",
            "near_count": 2,
        },
        {
            "name": "stage_9C_1near_3far",
            "near_count": 1,
        },
    ]

    def __init__(
        self,
        near_radius=2,
        far_min_distance=6,
    ):
        self.near_radius = near_radius
        self.far_min_distance = (
            far_min_distance
        )

        self.level = 0

    @property
    def name(self):

        return self.LEVELS[
            self.level
        ]["name"]

    def sample(self):

        config = self.LEVELS[
            self.level
        ]

        return make_stage9_scenario(
            near_count=
                config["near_count"],

            near_radius=
                self.near_radius,

            far_min_distance=
                self.far_min_distance,

            name=self.name,
        )

    def advance(self):

        if (
            self.level
            < len(self.LEVELS) - 1
        ):
            self.level += 1

            return True

        return False
