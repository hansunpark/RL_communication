import pytest

from env import CooperativeTransportEnv

from env.constants import (
    DOWN,
    RIGHT,
    STAY,
)

from evaluation.coordination_metrics import (
    CoordinationTracker,
)


def _env(
    agents,
    goal_x=8,
    goal_y=4,
    obstacles=(),
):
    """
    3x2 target을 (4,4)에 두고 에이전트 4명을 지정한 칸에 세운다.
    """

    env = CooperativeTransportEnv(
        num_agents=4,
    )

    env.reset(
        options={
            "scenario": {
                "name": "metric_test",
                "target": {
                    "x": 4,
                    "y": 4,
                    "width": 3,
                    "height": 2,
                },
                "goal_cells": [
                    [goal_x + dx, goal_y + dy]
                    for dy in range(2)
                    for dx in range(3)
                ],
                "obstacles": list(obstacles),
                "walls": [],
                "agent_positions": {
                    f"agent_{i}": list(position)
                    for i, position
                    in enumerate(agents)
                },
            }
        }
    )

    return env


FAR = [(0, 0), (11, 11)]


def test_full_face_and_synchronized_push():

    # goal이 오른쪽 -> 왼쪽 면 (3,4), (3,5)
    env = _env([(3, 4), (3, 5)] + FAR)

    tracker = CoordinationTracker()

    tracker.observe(
        env,
        {
            "agent_0": RIGHT,
            "agent_1": RIGHT,
            "agent_2": STAY,
            "agent_3": STAY,
        },
    )

    summary = tracker.summary()

    assert summary["rc"] == pytest.approx(1.0)
    assert summary["push_opportunities"] == 1
    assert summary["cpr"] == pytest.approx(1.0)


def test_one_pusher_out_of_sync():

    env = _env([(3, 4), (3, 5)] + FAR)

    tracker = CoordinationTracker()

    tracker.observe(
        env,
        {
            "agent_0": RIGHT,
            "agent_1": STAY,
        },
    )

    assert tracker.summary()["cpr"] == (
        pytest.approx(0.0)
    )


def test_half_face_has_no_push_opportunity():

    env = _env([(3, 4), (1, 1)] + FAR)

    tracker = CoordinationTracker()

    tracker.observe(env, {})

    summary = tracker.summary()

    assert summary["rc"] == pytest.approx(0.5)
    assert summary["push_opportunities"] == 0
    assert summary["cpr"] is None


def test_wrong_face_does_not_count():

    # 오른쪽 면(LEFT로 미는 자리)을 채웠지만 goal은 오른쪽
    env = _env([(7, 4), (7, 5)] + FAR)

    tracker = CoordinationTracker()

    tracker.observe(env, {})

    assert tracker.summary()["rc"] == (
        pytest.approx(0.0)
    )


def test_blocked_push_is_not_an_opportunity():

    env = _env(
        [(3, 4), (3, 5)] + FAR,
        obstacles=[
            {
                "x": 7,
                "y": 4,
                "width": 1,
                "height": 1,
            }
        ],
    )

    tracker = CoordinationTracker()

    tracker.observe(
        env,
        {
            "agent_0": RIGHT,
            "agent_1": RIGHT,
        },
    )

    summary = tracker.summary()

    assert summary["rc"] == pytest.approx(1.0)
    assert summary["push_opportunities"] == 0


def test_diagonal_goal_takes_best_direction():

    # goal이 오른쪽 아래 -> RIGHT(왼쪽 면 2칸)와 DOWN(위쪽 면 3칸)
    env = _env(
        [(4, 3), (5, 3), (6, 3), (0, 0)],
        goal_x=8,
        goal_y=8,
    )

    tracker = CoordinationTracker()

    tracker.observe(
        env,
        {
            "agent_0": DOWN,
            "agent_1": DOWN,
            "agent_2": DOWN,
        },
    )

    summary = tracker.summary()

    assert summary["rc"] == pytest.approx(1.0)
    assert summary["cpr"] == pytest.approx(1.0)
