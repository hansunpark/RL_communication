import numpy as np

from pettingzoo.test import parallel_api_test

from env.cooperative_transport_env import (
    CooperativeTransportEnv,
)

from env.objects import RectObject

from env.constants import (
    STAY,
    RIGHT,
    DOWN,
    HELP,
    NO_MESSAGE,
)


def no_message_action(direction):
    return np.array(
        [
            direction,
            NO_MESSAGE,
        ],
        dtype=np.int64,
    )


def test_object_geometry():

    obj = RectObject(
        object_id=0,
        x=4,
        y=3,
        width=3,
        height=2,
    )

    assert obj.cells() == {
        (4, 3),
        (5, 3),
        (6, 3),
        (4, 4),
        (5, 4),
        (6, 4),
    }

    assert obj.contact_cells(
        RIGHT
    ) == {
        (3, 3),
        (3, 4),
    }

    assert obj.moved_cells(
        RIGHT
    ) == {
        (5, 3),
        (6, 3),
        (7, 3),
        (5, 4),
        (6, 4),
        (7, 4),
    }


def test_successful_right_push():

    env = CooperativeTransportEnv(
        num_agents=2
    )

    env.reset(seed=0)

    obj = RectObject(
        object_id=0,
        x=3,
        y=3,
        width=3,
        height=2,
        is_target=True,
    )

    env.objects = [obj]

    env.goal_cells = {
        (x, y)
        for y in range(3, 5)
        for x in range(8, 11)
    }

    env.agent_positions = {
        "agent_0": (2, 3),
        "agent_1": (2, 4),
    }

    actions = {
        "agent_0":
            no_message_action(RIGHT),

        "agent_1":
            no_message_action(RIGHT),
    }

    env.step(actions)

    assert obj.x == 4
    assert obj.y == 3

    # pushers move into cells vacated by object
    assert env.agent_positions[
        "agent_0"
    ] == (3, 3)

    assert env.agent_positions[
        "agent_1"
    ] == (3, 4)


def test_push_fails_if_agent_missing():

    env = CooperativeTransportEnv(
        num_agents=2
    )

    env.reset(seed=0)

    obj = RectObject(
        object_id=0,
        x=3,
        y=3,
        width=3,
        height=2,
        is_target=True,
    )

    env.objects = [obj]

    env.agent_positions = {
        "agent_0": (2, 3),

        # 접촉면에 없음
        "agent_1": (1, 4),
    }

    actions = {
        "agent_0":
            no_message_action(RIGHT),

        "agent_1":
            no_message_action(STAY),
    }

    env.step(actions)

    assert obj.x == 3
    assert obj.y == 3


def test_push_fails_if_one_agent_chooses_wrong_action():

    env = CooperativeTransportEnv(
        num_agents=2
    )

    env.reset(seed=0)

    obj = RectObject(
        object_id=0,
        x=3,
        y=3,
        width=3,
        height=2,
        is_target=True,
    )

    env.objects = [obj]

    env.agent_positions = {
        "agent_0": (2, 3),
        "agent_1": (2, 4),
    }

    actions = {
        "agent_0":
            no_message_action(RIGHT),

        "agent_1":
            no_message_action(DOWN),
    }

    env.step(actions)

    assert obj.x == 3


def test_object_cannot_push_other_object():

    env = CooperativeTransportEnv(
        num_agents=2
    )

    env.reset(seed=0)

    object_a = RectObject(
        object_id=0,
        x=3,
        y=3,
        width=2,
        height=2,
        is_target=True,
    )

    object_b = RectObject(
        object_id=1,
        x=5,
        y=3,
        width=2,
        height=2,
        is_target=False,
    )

    env.objects = [
        object_a,
        object_b,
    ]

    env.agent_positions = {
        "agent_0": (2, 3),
        "agent_1": (2, 4),
    }

    actions = {
        "agent_0":
            no_message_action(RIGHT),

        "agent_1":
            no_message_action(RIGHT),
    }

    env.step(actions)

    # object_b가 있으므로 object_a push 실패
    assert object_a.x == 3

    # object_b도 움직이지 않음
    assert object_b.x == 5


def test_final_state_rule_allows_entering_vacated_cell():

    env = CooperativeTransportEnv(
        num_agents=3
    )

    env.reset(seed=0)

    obj = RectObject(
        object_id=0,
        x=3,
        y=3,
        width=2,
        height=2,
        is_target=True,
    )

    env.objects = [obj]

    env.agent_positions = {
        "agent_0": (2, 3),
        "agent_1": (2, 4),

        # push와 관계없는 agent
        "agent_2": (1, 3),
    }

    actions = {
        "agent_0":
            no_message_action(RIGHT),

        "agent_1":
            no_message_action(RIGHT),

        # agent_0이 비우는 cell로 이동
        "agent_2":
            no_message_action(RIGHT),
    }

    env.step(actions)

    assert obj.x == 4

    assert env.agent_positions[
        "agent_0"
    ] == (3, 3)

    assert env.agent_positions[
        "agent_1"
    ] == (3, 4)

    assert env.agent_positions[
        "agent_2"
    ] == (2, 3)


def test_diagonal_is_not_touching():

    env = CooperativeTransportEnv(
        num_agents=1
    )

    env.reset(seed=0)

    obj = RectObject(
        object_id=0,
        x=7,
        y=7,
        width=2,
        height=2,
        is_target=True,
    )

    env.objects = [obj]

    env.agent_positions = {
        "agent_0": (6, 6)
    }

    assert not env._agent_touching_object(
        "agent_0",
        obj,
    )


def test_contact_reveals_object_size():

    env = CooperativeTransportEnv(
        num_agents=1
    )

    env.reset(seed=0)

    obj = RectObject(
        object_id=0,
        x=4,
        y=4,
        width=3,
        height=2,
        is_target=True,
    )

    env.objects = [obj]

    env.agent_positions = {
        "agent_0": (3, 4)
    }

    observation = (
        env._get_observation(
            "agent_0"
        )
    )

    info = observation[
        "touching_objects"
    ][0]

    assert info[0] == 1
    assert info[1] == 3
    assert info[2] == 2
    assert info[3] == 1


def test_help_broadcast():

    env = CooperativeTransportEnv(
        num_agents=3
    )

    env.reset(seed=0)

    env.agent_positions = {
        "agent_0": (1, 1),
        "agent_1": (5, 1),
        "agent_2": (1, 5),
    }

    actions = {
        "agent_0": np.array(
            [
                STAY,
                HELP,
            ]
        ),

        "agent_1": np.array(
            [
                STAY,
                NO_MESSAGE,
            ]
        ),

        "agent_2": np.array(
            [
                STAY,
                NO_MESSAGE,
            ]
        ),
    }

    observations, *_ = env.step(
        actions
    )

    messages_1 = (
        observations[
            "agent_1"
        ]["messages"]
    )

    messages_2 = (
        observations[
            "agent_2"
        ]["messages"]
    )

    # agent_1 기준 agent_0:
    # (1,1) - (5,1) = (-4, 0)

    assert messages_1[0][0] == 1
    assert messages_1[0][1] == -4
    assert messages_1[0][2] == 0
    assert messages_1[0][3] == 0

    # agent_2 기준 agent_0:
    # (1,1) - (1,5) = (0, -4)

    assert messages_2[0][0] == 1
    assert messages_2[0][1] == 0
    assert messages_2[0][2] == -4


def test_parallel_api():

    env = CooperativeTransportEnv(
        num_agents=4
    )

    parallel_api_test(
        env,
        num_cycles=100,
    )


if __name__ == "__main__":

    test_object_geometry()

    test_successful_right_push()

    test_push_fails_if_agent_missing()

    test_push_fails_if_one_agent_chooses_wrong_action()

    test_object_cannot_push_other_object()

    test_final_state_rule_allows_entering_vacated_cell()

    test_diagonal_is_not_touching()

    test_contact_reveals_object_size()

    test_help_broadcast()

    test_parallel_api()

    print(
        "All environment tests passed."
    )