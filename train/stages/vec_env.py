import multiprocessing as mp

import numpy as np

from env import (
    CooperativeTransportEnv,
)

from env.constants import (
    TARGET_OBJECT,
)

from .common import (
    flatten_obs,
)


# ============================================================
# EnvPool: env 여러 개를 여러 프로세스에 나눠 reset/step
#
# comm_runner는 env 16개를 한 프로세스에서 순서대로 step했다. GPU로
# 정책과 update를 크게 batch하면(env 256개 = 1,024행) 정책 호출과
# update 비용은 거의 늘지 않지만, env를 한 프로세스에서 돌리는 시간이
# 병목이 된다 (env 16개 step 1.1 ms -> 256개면 약 18 ms).
#
# 시나리오 샘플링, 커리큘럼 판단, 버킷 기록은 지금처럼 메인 프로세스가
# 하고, worker는 받은 시나리오로 reset하고 행동을 받아 step한 뒤 평탄화한
# 관측만 돌려준다. num_workers=0이면 같은 인터페이스로 메인 프로세스에서
# 돈다 (기존 동작, 테스트용).
#
# 반환 형식 (env마다):
#   reset -> {"flat": [A, D], "visible": [A], "agents": [...]}
#   step  -> 위 + {"rewards": [A], "dones": [A], "success": bool,
#                  "episode_done": bool}
#   (episode_done이면 flat/visible은 끝난 상태의 관측이다. 메인이 곧바로
#    reset을 요청한다.)
# ============================================================


def apply_env_settings(
    env,
    settings,
):
    """
    레벨별 env 설정(help_penalty, max_steps)을 reset 직전에
    덮어쓴다. 두 값 모두 step()에서 매번 읽는 단순 속성이라
    reset 사이에 바꿔도 안전하다.
    """

    for key in (
        "help_penalty",
        "max_steps",
    ):
        if key in settings:
            setattr(
                env,
                key,
                settings[key],
            )


def _encode(
    env,
    space,
    observations,
):
    agents = list(env.agents) or list(
        observations
    )

    return {
        "flat": np.stack(
            [
                flatten_obs(
                    space,
                    observations[agent],
                )
                for agent in agents
            ]
        ),
        "visible": np.array(
            [
                bool(
                    (
                        observations[agent][
                            "local_map"
                        ]
                        == TARGET_OBJECT
                    ).any()
                )
                for agent in agents
            ]
        ),
        "agents": agents,
    }


class _LocalEnvs:
    """
    env 묶음을 다루는 실제 로직. 메인 프로세스(num_workers=0)와
    worker 프로세스가 같은 코드를 쓴다.
    """

    def __init__(
        self,
        env_kwargs,
        count,
    ):
        self.envs = [
            CooperativeTransportEnv(
                **env_kwargs
            )
            for _ in range(count)
        ]

        self.space = self.envs[
            0
        ].observation_space(
            self.envs[0].possible_agents[0]
        )

    def reset(
        self,
        requests,
    ):
        results = {}

        for (
            local_index,
            scenario,
            settings,
            seed,
        ) in requests:

            env = self.envs[local_index]

            apply_env_settings(
                env,
                settings,
            )

            observations, _ = env.reset(
                seed=seed,
                options={
                    "scenario": scenario
                },
            )

            results[local_index] = _encode(
                env,
                self.space,
                observations,
            )

        return results

    def step(
        self,
        actions,
    ):
        results = {}

        for local_index, env_actions in (
            actions.items()
        ):
            env = self.envs[local_index]

            agents = list(env.agents)

            (
                observations,
                rewards,
                terminations,
                truncations,
                infos,
            ) = env.step(env_actions)

            first = agents[0]

            result = _encode(
                env,
                self.space,
                observations,
            )

            result["agents"] = agents

            result["rewards"] = np.array(
                [
                    rewards[agent]
                    for agent in agents
                ],
                dtype=np.float32,
            )

            result["dones"] = np.array(
                [
                    bool(
                        terminations[agent]
                        or truncations[agent]
                    )
                    for agent in agents
                ]
            )

            result["success"] = bool(
                infos[first]["success"]
            )

            result["episode_done"] = (
                not env.agents
            )

            results[local_index] = result

        return results


def _worker(
    connection,
    env_kwargs,
    count,
    torch_threads,
):
    import torch

    torch.set_num_threads(
        torch_threads
    )

    local = _LocalEnvs(
        env_kwargs,
        count,
    )

    while True:

        command, payload = (
            connection.recv()
        )

        if command == "reset":
            connection.send(
                local.reset(payload)
            )

        elif command == "step":
            connection.send(
                local.step(payload)
            )

        elif command == "close":
            connection.close()

            return


class EnvPool:

    def __init__(
        self,
        env_kwargs,
        num_envs,
        num_workers=0,
    ):
        self.num_envs = num_envs

        self.num_workers = min(
            num_workers,
            num_envs,
        )

        probe = CooperativeTransportEnv(
            **env_kwargs
        )

        self.space = (
            probe.observation_space(
                probe.possible_agents[0]
            )
        )

        self.num_agents = (
            probe.num_agents
        )

        if self.num_workers == 0:

            self.local = _LocalEnvs(
                env_kwargs,
                num_envs,
            )

            self.owner = [
                (None, index)
                for index in range(num_envs)
            ]

            return

        self.local = None

        context = mp.get_context(
            "spawn"
        )

        self.connections = []

        self.processes = []

        # env를 worker에 고르게 나눈다.
        self.owner = []

        base, extra = divmod(
            num_envs,
            self.num_workers,
        )

        for worker in range(
            self.num_workers
        ):
            count = base + (
                1 if worker < extra else 0
            )

            parent, child = (
                context.Pipe()
            )

            process = context.Process(
                target=_worker,
                args=(
                    child,
                    env_kwargs,
                    count,
                    1,
                ),
                daemon=True,
            )

            process.start()

            child.close()

            self.connections.append(
                parent
            )

            self.processes.append(
                process
            )

            for local_index in range(
                count
            ):
                self.owner.append(
                    (worker, local_index)
                )

        self.env_index_of = {
            owner: env_index
            for env_index, owner
            in enumerate(self.owner)
        }

    def _dispatch(
        self,
        command,
        items,
    ):
        """
        items: {env_index: payload_item}. worker별로 묶어 보내고
        결과를 env_index 기준으로 모아 돌려준다.
        """

        if self.local is not None:

            local_payload = (
                items
                if command == "step"
                else [
                    (env_index, *item)
                    for env_index, item
                    in items.items()
                ]
            )

            return getattr(
                self.local,
                command,
            )(local_payload)

        per_worker = {}

        for env_index, item in (
            items.items()
        ):
            worker, local_index = (
                self.owner[env_index]
            )

            bucket = per_worker.setdefault(
                worker,
                {} if command == "step"
                else [],
            )

            if command == "step":
                bucket[local_index] = item
            else:
                bucket.append(
                    (local_index, *item)
                )

        for worker, payload in (
            per_worker.items()
        ):
            self.connections[
                worker
            ].send(
                (command, payload)
            )

        results = {}

        for worker in per_worker:

            reply = self.connections[
                worker
            ].recv()

            for (
                local_index,
                result,
            ) in reply.items():

                results[
                    self.env_index_of[
                        (worker, local_index)
                    ]
                ] = result

        return results

    def reset(
        self,
        requests,
    ):
        """
        requests: {env_index: (scenario, settings, seed)}
        """

        return self._dispatch(
            "reset",
            requests,
        )

    def step(
        self,
        actions,
    ):
        """
        actions: {env_index: {agent: np.array([move, comm])}}
        """

        return self._dispatch(
            "step",
            actions,
        )

    def close(self):

        if self.local is not None:
            return

        for connection in (
            self.connections
        ):
            try:
                connection.send(
                    ("close", None)
                )
            except (
                BrokenPipeError,
                OSError,
            ):
                pass

        for process in self.processes:
            process.join(timeout=5)

            # 첫 GRU 실행이 끝난 뒤 worker 하나가 남아 있었다.
            # 정상 종료하지 않은 worker는 강제로 끝낸다.
            if process.is_alive():
                process.terminate()
