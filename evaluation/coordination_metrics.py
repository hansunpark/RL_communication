from env.constants import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
)


# ============================================================
# 협동 지표: RC(모임)와 CPR(동기화된 밀기)
#
# 성공률만으로는 "성공하지 못한 episode에서 에이전트들이 얼마나
# 협동했는가"를 볼 수 없다. 통신 on/off 비교에서 두 지표를 같이 본다.
#
# 유효 방향 D_t: target을 goal 쪽으로 가깝게 만드는 방향
#   (target 좌상단과 goal 좌상단의 x/y 차이 부호로 정한다. 같은 행이면
#    가로 한 방향, 대각선이면 두 방향).
#
# RC (rally coverage): 매 step
#     C_t(o, d) = |F_t(o, d) ∩ P_t| / |F_t(o, d)|
#     RC_t = max_{d in D_t} C_t(o, d)
#   F_t(o, d)는 target을 d로 밀려면 서 있어야 하는 접촉 칸,
#   P_t는 에이전트가 차지한 칸. 유효 방향만 보므로 엉뚱한 면을 채운
#   것은 점수가 되지 않는다. RC는 유효 방향이 있는 모든 step의 평균.
#
# CPR (coordinated push rate):
#     CPR = (동기화된 유효 push 횟수) / (유효 push가 가능한 접촉 배치 횟수)
#   분모: 어떤 유효 방향 d에 대해 접촉 칸이 모두 에이전트로 차 있고,
#         target이 d로 움직일 공간(맵 안, 벽/다른 물체 없음)도 있는 step.
#   분자: 그 step에서 접촉 칸의 에이전트 전원이 d를 고른 경우.
#   (행동 선택 기준이다. 다른 에이전트와 칸이 겹쳐 실제 이동이
#    취소되는 경우도 "동기화된 시도"로 센다.)
# ============================================================


def useful_directions(
    target,
    goal_cells,
):
    goal_x = min(x for x, _ in goal_cells)
    goal_y = min(y for _, y in goal_cells)

    directions = []

    if goal_x > target.x:
        directions.append(RIGHT)
    elif goal_x < target.x:
        directions.append(LEFT)

    if goal_y > target.y:
        directions.append(DOWN)
    elif goal_y < target.y:
        directions.append(UP)

    return directions


def target_can_move(
    env,
    target,
    direction,
):
    blocked = set(env.walls)

    for obj in env.objects:
        if obj is not target:
            blocked |= obj.cells()

    return all(
        env._inside(cell)
        and cell not in blocked
        for cell in target.moved_cells(
            direction
        )
    )


class CoordinationTracker:

    def __init__(self):

        self.rc_sum = 0.0
        self.rc_steps = 0

        self.push_opportunities = 0
        self.coordinated_pushes = 0

    def observe(
        self,
        env,
        move_actions,
    ):
        """
        env.step 직전에 부른다. move_actions: {agent: 이동 행동}.
        """

        target = env._target_object()

        directions = useful_directions(
            target,
            env.goal_cells,
        )

        if not directions:
            return

        agent_at = {
            position: agent
            for agent, position
            in env.agent_positions.items()
        }

        best_coverage = 0.0

        ready = []

        for direction in directions:

            contact = target.contact_cells(
                direction
            )

            covered = [
                cell
                for cell in contact
                if cell in agent_at
            ]

            coverage = (
                len(covered) / len(contact)
            )

            best_coverage = max(
                best_coverage,
                coverage,
            )

            if (
                len(covered) == len(contact)
                and target_can_move(
                    env,
                    target,
                    direction,
                )
            ):
                ready.append(
                    (direction, contact)
                )

        self.rc_sum += best_coverage
        self.rc_steps += 1

        if not ready:
            return

        self.push_opportunities += 1

        if any(
            all(
                move_actions.get(
                    agent_at[cell]
                )
                == direction
                for cell in contact
            )
            for direction, contact in ready
        ):
            self.coordinated_pushes += 1

    def summary(self):

        return {
            "rc": (
                self.rc_sum / self.rc_steps
                if self.rc_steps
                else None
            ),
            "cpr": (
                self.coordinated_pushes
                / self.push_opportunities
                if self.push_opportunities
                else None
            ),
            "rc_steps": self.rc_steps,
            "push_opportunities":
                self.push_opportunities,
            "coordinated_pushes":
                self.coordinated_pushes,
        }
