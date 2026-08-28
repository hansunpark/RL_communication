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