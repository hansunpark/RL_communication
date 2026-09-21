import numpy as np

from pettingzoo.test import (
    parallel_api_test,
)

from env import (
    CooperativeTransportEnv,
    RectObject,
)

from env.constants import (
    STAY,
    DOWN,
    RIGHT,
    NO_MESSAGE,
    HELP,
)


def action(
    physical,
    message=NO_MESSAGE,
):

    return np.array(
        [
            physical,
            message,
        ],

        dtype=np.int64,
    )


def test_object_geometry():

    obj = RectObject(
        object_id=0,

        x=3,
        y=4,

        width=3,
        height=2,
    )

    assert obj.cells() == {
        (3, 4),
        (4, 4),
        (5, 4),
        (3, 5),
        (4, 5),
        (5, 5),
    }


def test_stage0_push():

    env = (
        CooperativeTransportEnv(
            curriculum_stage=0
        )
    )

    env.reset(
        seed=1
    )

    target = (
        env._target_object()
    )

    original_y = target.y

    actions = {
        agent:
            action(STAY)

        for agent
        in env.agents
    }

    actions[
        "agent_0"
    ] = action(DOWN)

    actions[
        "agent_1"
    ] = action(DOWN)

    env.step(
        actions
    )

    assert (
        target.y
        ==
        original_y + 1
    )


def test_missing_pusher_fails():

    env = (
        CooperativeTransportEnv(
            curriculum_stage=0
        )
    )

    env.reset(
        seed=1
    )

    target = (
        env._target_object()
    )

    original_y = target.y

    actions = {
        agent:
            action(STAY)

        for agent
        in env.agents
    }

    actions[
        "agent_0"
    ] = action(DOWN)

    env.step(
        actions
    )

    assert (
        target.y
        ==
        original_y
    )


def test_help_broadcast():

    env = (
        CooperativeTransportEnv(
            curriculum_stage=0
        )
    )

    env.reset(
        seed=1
    )

    actions = {
        agent:
            action(STAY)

        for agent
        in env.agents
    }

    actions[
        "agent_0"
    ] = action(
        STAY,
        HELP,
    )

    (
        observations,
        _,
        _,
        _,
        _,
    ) = env.step(
        actions
    )

    for agent in (
        "agent_1",
        "agent_2",
        "agent_3",
    ):

        messages = (
            observations[
                agent
            ][
                "messages"
            ]
        )

        assert (
            messages[:, 0].sum()
            >= 1
        )


def test_spawn_split():

    env = (
        CooperativeTransportEnv(
            curriculum_stage=0
        )
    )

    scenario = {
        "name": "split_test",

        "target": {
            "x": 4,
            "y": 4,
            "width": 3,
            "height": 2,
        },

        "goal_cells": [
            [4, 9],
            [5, 9],
            [6, 9],
        ],

        "obstacles": [],

        "spawn_mode": {
            "type": "split",
            "near_count": 1,
            "near_radius": 2,
            "far_min_distance": 6,
        },
    }

    env.reset(
        seed=7,
        options={
            "scenario": scenario
        },
    )

    target = env._target_object()

    reference_cells = (
        target.cells()
    )

    def distance(pos):

        x, y = pos

        return min(
            abs(x - rx) + abs(y - ry)
            for rx, ry in reference_cells
        )

    positions = list(
        env.agent_positions.values()
    )

    # No overlaps.
    assert len(set(positions)) == len(
        positions
    )

    distances = sorted(
        distance(pos)
        for pos in positions
    )

    # 1 near agent within radius 2.
    assert distances[0] <= 2

    # remaining 3 agents at least
    # far_min_distance away.
    for d in distances[1:]:
        assert d >= 6


def test_parallel_api():

    env = (
        CooperativeTransportEnv(
            curriculum_stage=0
        )
    )

    parallel_api_test(
        env,
        num_cycles=100,
    )