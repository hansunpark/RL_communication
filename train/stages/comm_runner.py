import json
import os
import random
import time

from collections import deque
from dataclasses import (
    dataclass,
    field,
)

import numpy as np
import torch

from gymnasium.spaces import (
    flatdim,
)

from torch.utils.tensorboard import (
    SummaryWriter,
)

from env import (
    CooperativeTransportEnv,
)

from env.constants import (
    HELP,
    NO_MESSAGE,
    TARGET_OBJECT,
)

from ..ppo import PPO

from ..recurrent_ppo import (
    RecurrentPPO,
)

from ..recurrent_buffer import (
    RecurrentRolloutBuffer,
)

from ..rollout_buffer import (
    MultiAgentRolloutBuffer,
)

from .bucketed_curriculum import (
    BucketedCurriculumManager,
)

from .common import (
    set_seed,
    flatten_obs,
)

from .vec_env import (
    EnvPool,
)

from evaluation.coordination_metrics import (
    CoordinationTracker,
)


# ============================================================
# 통신 학습용 공용 러너 (벡터화 env + 레벨 커리큘럼)
#
# stage_8_comm(8B1부터 통신 켜고 이어가기)과 comm_scratch(처음부터
# 통신 켠 새 커리큘럼)가 같이 쓴다. 기존 stage별 train.py와 다른 점:
#
# 1. env N개를 동시에 돌리고 모든 env x agent의 관측을 forward
#    한 번으로 처리한다. 기존 루프는 agent마다 batch=1 forward를
#    해서 초당 ~870 env-step에 묶여 있었다 (env 자체는 ~14k/s).
# 2. comm_action을 실제로 env에 보낸다 (comm_enabled=True).
# 3. 레벨이 mastered될 때마다 같은 체크포인트로 통신 on/off
#    ablation 평가를 돌려서 결과를 JSON으로 남긴다
#    (docs/curriculum_plan.md의 "검증 필수" 항목).
# 4. 레벨별 최대 episode를 넘기면 멈추고 상태를 남긴다 (정체된
#    레벨에서 계산을 무한히 태우지 않기 위함).
# 5. _latest.pt + _latest.json으로 중단 후 재개할 수 있다.
#
# Sampler 인터페이스 (duck typing):
#   level, name, LEVEL_NAMES, sample(), bucket_for(scenario),
#   advance(), env_settings() -> {"help_penalty": .., "max_steps": ..}
# ============================================================


@dataclass
class RunConfig:

    run_name: str

    seed: int = 42

    num_envs: int = 16

    # env 하나당 rollout 길이. transition 수는
    # num_envs * rollout_steps * num_agents.
    rollout_steps: int = 128

    gamma: float = 0.99

    gae_lambda: float = 0.95

    learning_rate: float = 3e-4

    hidden_dim: int = 256

    clip_coef: float = 0.2

    value_coef: float = 0.5

    entropy_coef: float = 0.01

    comm_entropy_coef: float = 0.01

    # PPO KL 조기 종료 (None이면 끔). PPO.target_kl 참고.
    target_kl: float = None

    update_epochs: int = 4

    minibatch_size: int = 512

    success_threshold: float = 0.85

    window_size: int = 100

    min_episodes_per_level: int = 1000

    max_episodes_per_level: int = 60000

    start_checkpoint: str = None

    comm_enabled: bool = True

    ablation_episodes: int = 300

    torch_threads: int = 4

    # warm-start 직후 처음 이 횟수만큼의 update는 critic head만
    # 학습한다 (PPO.update(update_policy=False) 참고). 재개(--resume)
    # 시에는 적용하지 않는다.
    critic_warmup_updates: int = 0

    # 이 확률로 이미 통과한 이전 레벨(이 실행 안에서)의 시나리오를
    # 뽑아 학습한다 (mastery 통계에는 넣지 않는다).
    #
    # 도입 이유: comm_scratch S3(크기 랜덤)에서 실패가 대부분이
    # 되자, S2에서 85~97%였던 작은 target까지 25~39%로 무너졌다.
    # Stage 8B1/stage8_comm 1차 실행과 같은 "실패만 있는 구간에서
    # 기존 능력을 잃는" 패턴이라, 이미 풀 수 있는 문제를 섞어서
    # 성공 신호를 유지한다.
    rehearsal_prob: float = 0.0

    # 재개가 아닌 새 실행에서 시작할 레벨 인덱스. 예: 이미 통과한
    # 레벨의 _mastered.pt에서 이어갈 때.
    start_level: int = 0

    # True면 시작(또는 재개)한 레벨을 통과하는 즉시 멈춘다. 멈추기
    # 전에 다음 레벨로 넘어간 상태를 _latest에 저장하므로, 나중에
    # --resume하면 다음 레벨부터 이어간다.
    stop_after_current_level: bool = False

    # True면 GRU 메모리를 단 정책(RecurrentPPO)을 쓴다. MLP
    # 체크포인트에서 시작해도 처음에는 MLP와 똑같이 행동한다
    # (train/network.py RecurrentActorCritic 참고).
    recurrent: bool = False

    rnn_hidden_dim: int = 128

    # truncated BPTT 조각 길이. rollout_steps가 이 값으로 나누어
    # 떨어져야 하고, 아니면 rollout 전체를 한 조각으로 쓴다.
    bptt_length: int = 32

    # env를 나눠 돌릴 worker 프로세스 수. 0이면 메인 프로세스에서
    # 순서대로 돈다 (기존 동작). GPU로 큰 batch를 쓸 때 env step이
    # 병목이 되므로 env를 늘리면서 worker도 늘린다 (vec_env.py 참고).
    num_workers: int = 0

    # "auto"면 CUDA가 있으면 GPU, 없으면 CPU.
    device: str = "auto"

    print_interval_episodes: int = 200

    env_kwargs: dict = field(
        default_factory=dict
    )


def _fmt(value):

    return (
        "-"
        if value is None
        else f"{value:.3f}"
    )


def _target_visible(observation):

    return bool(
        (
            observation["local_map"]
            == TARGET_OBJECT
        ).any()
    )


def _safe_save(
    ppo,
    path,
    retries=20,
):
    """
    임시 파일에 저장한 뒤 os.replace로 교체한다. 실패하면 잠깐
    기다렸다가 다시 시도한다.

    comm_scratch 2부가 _latest.pt를 덮어쓰다가 Windows 오류
    1224(다른 프로세스가 매핑해 둔 파일)로 죽었다. 백신 검사나
    다른 도구가 파일을 잠깐 열고 있는 경우라, 저장 한 번 실패로
    학습 전체가 멈추지 않게 한다.
    """

    temp_path = f"{path}.tmp"

    for attempt in range(retries):

        try:
            ppo.save(temp_path)

            os.replace(temp_path, path)

            return

        except (
            OSError,
            RuntimeError,
        ) as error:

            print(
                f"save retry {attempt + 1}/"
                f"{retries} for {path}: "
                f"{error}",
                flush=True,
            )

            time.sleep(1.0)

    raise RuntimeError(
        f"Could not save {path}"
    )


def _apply_env_settings(
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


def _reset_with_scenario(
    env,
    sampler,
    seed=None,
    scenario=None,
    max_tries=50,
):
    """
    split spawn은 target 위치에 따라 far 후보 칸이 모자라
    RuntimeError를 낼 수 있다. 그런 시나리오는 버리고 다시
    뽑는다 (scenario를 직접 준 경우는 재시도하지 않는다).
    """

    last_error = None

    for _ in range(max_tries):

        current = (
            scenario
            if scenario is not None
            else sampler.sample()
        )

        try:
            observations, _ = env.reset(
                seed=seed,
                options={
                    "scenario": current
                },
            )

            return current, observations

        except RuntimeError as error:

            last_error = error

            if scenario is not None:
                raise

            seed = None

    raise RuntimeError(
        f"Could not build a valid "
        f"scenario for "
        f"{sampler.name}: {last_error}"
    )


# ============================================================
# 평가 (통신 on/off ablation)
# ============================================================

def evaluate_level(
    ppo,
    sampler,
    env_kwargs,
    episodes,
    comm_enabled,
    seed,
    num_envs=16,
):
    """
    현재 레벨에서 episodes개를 돌려 성공률을 잰다. 통신 on/off를
    같은 seed로 부르면 동일한 시나리오 목록 + 동일한 env seed로
    비교된다 (paired 비교). 학습과 같은 stochastic 정책을 쓴다.
    """

    python_state = random.getstate()

    random.seed(seed)

    scenarios = []

    probe_env = CooperativeTransportEnv(
        **env_kwargs
    )

    _apply_env_settings(
        probe_env,
        sampler.env_settings(),
    )

    # split spawn이 불가능한 시나리오를 미리 걸러서 on/off가
    # 정확히 같은 목록을 쓰게 한다.
    while len(scenarios) < episodes:

        scenario, _ = _reset_with_scenario(
            probe_env,
            sampler,
        )

        scenarios.append(scenario)

    random.setstate(python_state)

    torch_state = torch.get_rng_state()

    torch.manual_seed(seed)

    envs = [
        CooperativeTransportEnv(
            **env_kwargs
        )
        for _ in range(num_envs)
    ]

    for env in envs:
        _apply_env_settings(
            env,
            sampler.env_settings(),
        )

    successes = []
    lengths = []
    help_steps = 0
    agent_steps = 0

    # RC/CPR (evaluation/coordination_metrics.py)
    tracker = CoordinationTracker()

    next_index = 0

    active = {}

    recurrent = getattr(
        ppo,
        "recurrent",
        False,
    )

    def start(env_index):

        nonlocal next_index

        if next_index >= len(scenarios):
            return

        scenario_index = next_index

        next_index += 1

        _, observations = (
            _reset_with_scenario(
                envs[env_index],
                sampler,
                seed=seed + scenario_index,
                scenario=scenarios[
                    scenario_index
                ],
            )
        )

        active[env_index] = {
            "obs": observations,
            "length": 0,
            "first_opportunity": None,
        }

        if recurrent:
            active[env_index][
                "hidden"
            ] = ppo.initial_hidden(
                len(envs[env_index].agents)
            )

    for env_index in range(num_envs):
        start(env_index)

    space = envs[0].observation_space(
        envs[0].possible_agents[0]
    )

    while active:

        keys = []
        flats = []

        for env_index, state in (
            active.items()
        ):
            for agent in (
                envs[env_index].agents
            ):
                keys.append(
                    (env_index, agent)
                )

                flats.append(
                    flatten_obs(
                        space,
                        state["obs"][agent],
                    )
                )

        if recurrent:

            env_order = list(active)

            hidden = torch.cat(
                [
                    active[env_index][
                        "hidden"
                    ]
                    for env_index
                    in env_order
                ]
            )

            (
                actions,
                comm_actions,
                _,
                _,
                new_hidden,
            ) = ppo.act_batch_recurrent(
                np.stack(flats),
                hidden,
                np.zeros(
                    len(flats),
                    dtype=np.float32,
                ),
            )

            offset = 0

            for env_index in env_order:

                count = len(
                    envs[env_index].agents
                )

                active[env_index][
                    "hidden"
                ] = new_hidden[
                    offset:
                    offset + count
                ]

                offset += count

        else:
            (
                actions,
                comm_actions,
                _,
                _,
            ) = ppo.act_batch(
                np.stack(flats)
            )

        per_env = {}

        for (
            (env_index, agent),
            action,
            comm_action,
        ) in zip(
            keys,
            actions,
            comm_actions,
        ):
            sent = (
                int(comm_action)
                if comm_enabled
                else NO_MESSAGE
            )

            help_steps += int(
                sent == HELP
            )

            agent_steps += 1

            per_env.setdefault(
                env_index,
                {},
            )[agent] = np.array(
                [int(action), sent],
                dtype=np.int64,
            )

        for env_index, env_actions in (
            per_env.items()
        ):
            env = envs[env_index]

            state = active[env_index]

            opportunity = tracker.observe(
                env,
                {
                    agent: int(action[0])
                    for agent, action
                    in env_actions.items()
                },
            )

            if (
                opportunity
                and state[
                    "first_opportunity"
                ]
                is None
            ):
                state[
                    "first_opportunity"
                ] = state["length"]

            (
                observations,
                _,
                _,
                _,
                infos,
            ) = env.step(env_actions)

            state = active[env_index]

            state["obs"] = observations

            state["length"] += 1

            if not env.agents:

                first = next(iter(infos))

                successes.append(
                    bool(
                        infos[first][
                            "success"
                        ]
                    )
                )

                lengths.append(
                    state["length"]
                )

                tracker.end_episode(
                    state[
                        "first_opportunity"
                    ],
                    state["length"],
                )

                del active[env_index]

                start(env_index)

    torch.set_rng_state(torch_state)

    return {
        "episodes": len(successes),

        "success_rate": float(
            np.mean(successes)
        ),

        "avg_length": float(
            np.mean(lengths)
        ),

        "help_rate": float(
            help_steps
            / max(1, agent_steps)
        ),

        **tracker.summary(),
    }


def run_ablation(
    ppo,
    sampler,
    config,
    tag,
):

    common = dict(
        ppo=ppo,
        sampler=sampler,
        env_kwargs=config.env_kwargs,
        episodes=config.ablation_episodes,
        seed=config.seed * 7919 + 17,
        # 평가는 메인 프로세스에서 돈다. 학습 env가 수백 개여도
        # 평가 episode는 수백 개라 이 정도면 충분하다.
        num_envs=min(
            config.num_envs,
            32,
        ),
    )

    comm_on = evaluate_level(
        comm_enabled=True,
        **common,
    )

    comm_off = evaluate_level(
        comm_enabled=False,
        **common,
    )

    result = {
        "run": config.run_name,
        "level": sampler.name,
        "tag": tag,
        "comm_on": comm_on,
        "comm_off": comm_off,
        "success_gap": (
            comm_on["success_rate"]
            - comm_off["success_rate"]
        ),
    }

    out_dir = os.path.join(
        "evaluation_results",
        config.run_name,
    )

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    with open(
        os.path.join(
            out_dir,
            f"ablation_{sampler.name}"
            f"_{tag}.json",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    return result


# ============================================================
# 학습
# ============================================================

def train(
    config,
    sampler,
    resume=False,
):

    set_seed(config.seed)

    torch.set_num_threads(
        config.torch_threads
    )

    os.makedirs(
        "checkpoints",
        exist_ok=True,
    )

    os.makedirs(
        "logs",
        exist_ok=True,
    )

    prefix = os.path.join(
        "checkpoints",
        config.run_name,
    )

    latest_path = (
        f"{prefix}_latest.pt"
    )

    latest_meta_path = (
        f"{prefix}_latest.json"
    )

    status_path = os.path.join(
        "logs",
        f"{config.run_name}_status.json",
    )

    history = []

    device = (
        (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        if config.device == "auto"
        else config.device
    )

    print(f"Device: {device}")

    pool = EnvPool(
        config.env_kwargs,
        config.num_envs,
        num_workers=config.num_workers,
    )

    # 시나리오 검증용 (split spawn이 가능한 배치인지 메인에서
    # 미리 확인한다. 가능 여부는 기하학만으로 정해진다).
    probe_env = CooperativeTransportEnv(
        **config.env_kwargs
    )

    space = pool.space

    obs_dim = flatdim(space)

    print(
        f"Env pool: {config.num_envs} envs "
        f"on {pool.num_workers} workers"
    )

    ppo_class = PPO

    ppo_extra = {}

    if config.recurrent:

        ppo_class = RecurrentPPO

        ppo_extra = {
            "rnn_hidden_dim":
                config.rnn_hidden_dim,
            "bptt_length":
                config.bptt_length,
        }

    ppo = ppo_class(
        obs_dim=obs_dim,
        num_actions=5,
        **ppo_extra,
        learning_rate=
            config.learning_rate,
        hidden_dim=config.hidden_dim,
        clip_coef=config.clip_coef,
        value_coef=config.value_coef,
        entropy_coef=
            config.entropy_coef,
        comm_entropy_coef=
            config.comm_entropy_coef,
        target_kl=config.target_kl,
        update_epochs=
            config.update_epochs,
        minibatch_size=
            config.minibatch_size,
        device=device,
    )

    global_step = 0

    total_episodes = 0

    warmup_remaining = (
        config.critic_warmup_updates
    )

    level_episodes = 0

    if (
        resume
        and os.path.exists(latest_path)
    ):

        ppo.load(
            latest_path,
            load_optimizer=True,
        )

        with open(
            latest_meta_path,
            encoding="utf-8",
        ) as file:
            meta = json.load(file)

        sampler.level = meta["level"]

        warmup_remaining = 0

        global_step = meta["global_step"]

        total_episodes = (
            meta["total_episodes"]
        )

        history = meta.get(
            "history",
            [],
        )

        print(
            f"Resumed from {latest_path} "
            f"at {sampler.name}"
        )

    else:
        sampler.level = config.start_level

        if config.start_checkpoint:

            print(
                "Loading start checkpoint: "
                f"{config.start_checkpoint}"
            )

            ppo.load(
                config.start_checkpoint,
                load_optimizer=True,
            )

    # 체크포인트의 optimizer state를 불러오면 저장 당시 lr이
    # 복원되므로 설정값으로 다시 맞춘다.
    for group in (
        ppo.optimizer.param_groups
    ):
        group["lr"] = (
            config.learning_rate
        )

    print(
        f"Observation dim: {obs_dim}"
    )

    stop_after_level = (
        sampler.level
        if config.stop_after_current_level
        else None
    )

    if stop_after_level is not None:
        print(
            "Will stop after mastering "
            f"{sampler.name}",
            flush=True,
        )

    print(
        f"Run: {config.run_name} "
        f"envs={config.num_envs} "
        f"comm_enabled="
        f"{config.comm_enabled} "
        f"recurrent={config.recurrent}"
    )

    writer = SummaryWriter(
        os.path.join(
            "runs",
            config.run_name,
        )
    )

    curriculum = (
        BucketedCurriculumManager(
            threshold=
                config.success_threshold,
            window_size=
                config.window_size,
            min_total_episodes=
                config.min_episodes_per_level,
        )
    )

    buffer = (
        RecurrentRolloutBuffer(
            device=device
        )
        if config.recurrent
        else MultiAgentRolloutBuffer(
            device=device
        )
    )

    # GRU hidden state. 행 순서는 정책 호출 때의 keys 순서
    # (env 순서 x 각 env의 에이전트 순서)이고, 모든 env가 매 step
    # 에이전트 전원을 움직이므로 rollout 내내 고정된다.
    num_rows = (
        config.num_envs
        * pool.num_agents
    )

    hidden = (
        ppo.initial_hidden(num_rows)
        if config.recurrent
        else None
    )

    # 1이면 다음 step 직전에 그 행의 hidden을 0으로 만든다
    # (새 episode 시작).
    episode_starts = np.ones(
        num_rows,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # env별 episode 상태
    # --------------------------------------------------------

    states = [None] * config.num_envs

    # start_episode가 뽑은 시나리오. env 루프가 끝난 뒤
    # flush_resets()에서 한꺼번에(worker별 병렬로) reset한다.
    pending_resets = {}

    def start_episode(env_index):

        current_level = sampler.level

        rehearsal = (
            current_level > 0
            and random.random()
            < config.rehearsal_prob
        )

        if rehearsal:
            sampler.level = random.randrange(
                current_level
            )

        try:
            settings = (
                sampler.env_settings()
            )

            _apply_env_settings(
                probe_env,
                settings,
            )

            scenario, _ = (
                _reset_with_scenario(
                    probe_env,
                    sampler,
                )
            )

            bucket = sampler.bucket_for(
                scenario
            )

        finally:
            sampler.level = current_level

        pending_resets[env_index] = (
            scenario,
            settings,
            (
                config.seed * 1000
                + env_index
                if states[env_index]
                is None
                else None
            ),
        )

        states[env_index] = {
            "flat": None,
            "visible": None,
            "agents": None,
            "level": current_level,
            "rehearsal": rehearsal,
            "bucket": bucket,
            "return": 0.0,
            "length": 0,
            "help": 0,
            "help_visible": 0,
            "help_hidden": 0,
            "steps_visible": 0,
            "steps_hidden": 0,
            "agent_steps": 0,
        }

    def flush_resets():

        if not pending_resets:
            return

        results = pool.reset(
            dict(pending_resets)
        )

        pending_resets.clear()

        for env_index, result in (
            results.items()
        ):
            states[env_index].update(
                flat=result["flat"],
                visible=result["visible"],
                agents=result["agents"],
            )

    for env_index in range(
        config.num_envs
    ):
        start_episode(env_index)

    flush_resets()

    # 다음 forward에서 value가 채워질 transition.
    # key: (env_index, agent)
    pending = {}

    length_window = deque(maxlen=200)
    return_window = deque(maxlen=200)
    help_window = deque(maxlen=200)
    help_visible_window = deque(
        maxlen=200
    )
    help_hidden_window = deque(
        maxlen=200
    )

    last_metrics = {}

    start_time = time.time()

    stop_reason = None

    def write_status(extra=None):

        status = {
            "run": config.run_name,
            "level": sampler.level,
            "level_name": sampler.name,
            "num_levels": len(
                sampler.LEVEL_NAMES
            ),
            "level_episodes":
                level_episodes,
            "total_episodes":
                total_episodes,
            "global_step": global_step,
            "elapsed_min": round(
                (
                    time.time()
                    - start_time
                )
                / 60.0,
                1,
            ),
            "avg_len": float(
                np.mean(length_window)
            ) if length_window else None,
            "avg_return": float(
                np.mean(return_window)
            ) if return_window else None,
            "help_rate": float(
                np.mean(help_window)
            ) if help_window else None,
            "buckets":
                curriculum.status(),
            "last_update": last_metrics,
            "history": history,
            "stop_reason": stop_reason,
        }

        if extra:
            status.update(extra)

        with open(
            status_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                status,
                file,
                indent=2,
            )

    def save_latest():

        _safe_save(ppo, latest_path)

        with open(
            latest_meta_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "level": sampler.level,
                    "global_step":
                        global_step,
                    "total_episodes":
                        total_episodes,
                    "history": history,
                },
                file,
                indent=2,
            )

    def collect_flats():
        """
        모든 env x agent의 (키 목록, 평탄화한 관측 [N, D]).
        """

        keys = [
            (env_index, agent)
            for env_index, state
            in enumerate(states)
            for agent in state["agents"]
        ]

        return keys, np.concatenate(
            [
                state["flat"]
                for state in states
            ]
        )

    rollout_step = 0

    while stop_reason is None:

        # ====================================================
        # Policy (모든 env x agent를 한 번에)
        # ====================================================

        keys, flat_batch = (
            collect_flats()
        )

        if config.recurrent:

            assert len(keys) == num_rows

            hidden_in = hidden

            (
                actions,
                comm_actions,
                log_probs,
                values,
                new_hidden,
            ) = ppo.act_batch_recurrent(
                flat_batch,
                hidden,
                episode_starts,
            )

            step_rewards = np.zeros(
                num_rows,
                dtype=np.float32,
            )

            step_dones = np.zeros(
                num_rows,
                dtype=np.float32,
            )

        else:
            (
                actions,
                comm_actions,
                log_probs,
                values,
            ) = ppo.act_batch(flat_batch)

        # 직전 step의 transition에 next_value를 채운다
        # (지금 관측의 value = 그 transition의 next_value).
        # recurrent 경로는 pending을 쓰지 않는다.
        for index, key in enumerate(keys):

            if key in pending:

                transition = pending.pop(
                    key
                )

                transition[
                    "next_value"
                ] = float(values[index])

                buffer.add(**transition)

        env_actions = {}

        visible_flat = np.concatenate(
            [
                state["visible"]
                for state in states
            ]
        )

        for index, (
            env_index,
            agent,
        ) in enumerate(keys):

            sent = (
                int(comm_actions[index])
                if config.comm_enabled
                else NO_MESSAGE
            )

            env_actions.setdefault(
                env_index,
                {},
            )[agent] = np.array(
                [
                    int(actions[index]),
                    sent,
                ],
                dtype=np.int64,
            )

            state = states[env_index]

            visible = bool(
                visible_flat[index]
            )

            state["agent_steps"] += 1

            if visible:
                state["steps_visible"] += 1
            else:
                state["steps_hidden"] += 1

            if sent == HELP:

                state["help"] += 1

                if visible:
                    state[
                        "help_visible"
                    ] += 1
                else:
                    state[
                        "help_hidden"
                    ] += 1

        # ====================================================
        # Env step
        # ====================================================

        key_index = {
            key: index
            for index, key
            in enumerate(keys)
        }

        step_results = pool.step(
            env_actions
        )

        for env_index in range(
            config.num_envs
        ):
            state = states[env_index]

            result = step_results[
                env_index
            ]

            episode_done = result[
                "episode_done"
            ]

            for agent_position, agent in (
                enumerate(result["agents"])
            ):
                index = key_index[
                    (env_index, agent)
                ]

                reward = result["rewards"][
                    agent_position
                ]

                done = bool(
                    result["dones"][
                        agent_position
                    ]
                )

                transition = {
                    "agent":
                        f"e{env_index}"
                        f"_{agent}",
                    "observation":
                        flat_batch[index],
                    "action":
                        int(actions[index]),
                    "comm_action":
                        int(
                            comm_actions[
                                index
                            ]
                        ),
                    "log_prob":
                        float(
                            log_probs[index]
                        ),
                    "reward":
                        float(reward),
                    "done": done,
                    "value":
                        float(values[index]),
                    "next_value": 0.0,
                }

                if config.recurrent:

                    step_rewards[index] = (
                        float(reward)
                    )

                    step_dones[index] = (
                        float(done)
                    )

                elif done:
                    buffer.add(**transition)

                else:
                    pending[
                        (env_index, agent)
                    ] = transition

                global_step += 1

            state["return"] += float(
                result["rewards"][0]
            )

            state["length"] += 1

            state["flat"] = result["flat"]

            state["visible"] = (
                result["visible"]
            )

            state["agents"] = (
                result["agents"]
            )

            if not episode_done:
                continue

            # ------------------------------------------------
            # Episode 종료
            # ------------------------------------------------

            success = result["success"]

            total_episodes += 1

            # 레벨이 바뀌기 전에 시작된 episode는 새 레벨의
            # 통계에 섞지 않는다.
            if (
                state["level"]
                == sampler.level
                and not state["rehearsal"]
            ):
                level_episodes += 1

                curriculum.record(
                    state["bucket"],
                    success,
                )

                length_window.append(
                    state["length"]
                )

                return_window.append(
                    state["return"]
                )

                help_window.append(
                    state["help"]
                    / max(
                        1,
                        state[
                            "agent_steps"
                        ],
                    )
                )

                if state["steps_visible"]:
                    help_visible_window.append(
                        state[
                            "help_visible"
                        ]
                        / state[
                            "steps_visible"
                        ]
                    )

                if state["steps_hidden"]:
                    help_hidden_window.append(
                        state[
                            "help_hidden"
                        ]
                        / state[
                            "steps_hidden"
                        ]
                    )

                if (
                    level_episodes
                    % config
                    .print_interval_episodes
                    == 0
                ):
                    # 버킷이 최대 30개라 콘솔에는 준비된 버킷
                    # 수와 가장 낮은 버킷 3개만 찍는다 (전체는
                    # status JSON에 남는다).
                    bucket_status = (
                        curriculum.status()
                    )

                    ready_count = sum(
                        info["ready"]
                        for info
                        in bucket_status.values()
                    )

                    worst = sorted(
                        bucket_status.items(),
                        key=lambda item: (
                            item[1][
                                "success_rate"
                            ]
                        ),
                    )[:3]

                    rates = (
                        f"ready {ready_count}/"
                        f"{len(bucket_status)} "
                        "worst: "
                        + " ".join(
                            f"{bucket}="
                            f"{info['success_rate']:.2f}"
                            f"({info['n']})"
                            for bucket, info
                            in worst
                        )
                    )

                    elapsed = (
                        time.time()
                        - start_time
                    ) / 60.0

                    print(
                        f"[{elapsed:6.1f}m "
                        f"ep {total_episodes:6d}] "
                        f"{sampler.name} "
                        f"lvl_ep={level_episodes} "
                        f"avg_len="
                        f"{np.mean(length_window):.1f} "
                        f"return="
                        f"{np.mean(return_window):.2f} "
                        f"help="
                        f"{np.mean(help_window):.3f} "
                        f"| {rates}",
                        flush=True,
                    )

                    for (
                        bucket,
                        info,
                    ) in (
                        curriculum
                        .status()
                        .items()
                    ):
                        writer.add_scalar(
                            f"success_rate/"
                            f"{sampler.name}/"
                            f"{bucket}",
                            info["success_rate"],
                            total_episodes,
                        )

                    writer.add_scalar(
                        "episode/avg_length",
                        np.mean(length_window),
                        total_episodes,
                    )

                    writer.add_scalar(
                        "episode/avg_return",
                        np.mean(return_window),
                        total_episodes,
                    )

                    writer.add_scalar(
                        "comm/help_rate",
                        np.mean(help_window),
                        total_episodes,
                    )

                    if help_visible_window:
                        writer.add_scalar(
                            "comm/help_rate_target_visible",
                            np.mean(
                                help_visible_window
                            ),
                            total_episodes,
                        )

                    if help_hidden_window:
                        writer.add_scalar(
                            "comm/help_rate_target_hidden",
                            np.mean(
                                help_hidden_window
                            ),
                            total_episodes,
                        )

                    writer.add_scalar(
                        "curriculum/level",
                        sampler.level,
                        total_episodes,
                    )

                    write_status(
                        {
                            "help_rate_target_visible":
                                float(
                                    np.mean(
                                        help_visible_window
                                    )
                                )
                                if help_visible_window
                                else None,
                            "help_rate_target_hidden":
                                float(
                                    np.mean(
                                        help_hidden_window
                                    )
                                )
                                if help_hidden_window
                                else None,
                        }
                    )

            # ------------------------------------------------
            # 레벨 전환 / 정체 판정
            # ------------------------------------------------

            if curriculum.mastered():

                level_name = sampler.name

                mastered_level = sampler.level

                print()
                print("=" * 70)
                print(
                    f"{level_name} mastered "
                    f"after {level_episodes} "
                    f"episodes"
                )

                for (
                    bucket,
                    info,
                ) in sorted(
                    curriculum
                    .status()
                    .items()
                ):
                    print(
                        f"  {bucket:<28} "
                        f"n={info['n']:4d} "
                        f"success="
                        f"{info['success_rate']:.3f}"
                    )

                _safe_save(
                    ppo,
                    f"{prefix}_"
                    f"{level_name}"
                    f"_mastered.pt"
                )

                ablation = run_ablation(
                    ppo,
                    sampler,
                    config,
                    tag="mastered",
                )

                print(
                    "  ablation: comm_on="
                    f"{ablation['comm_on']['success_rate']:.3f} "
                    "comm_off="
                    f"{ablation['comm_off']['success_rate']:.3f} "
                    "gap="
                    f"{ablation['success_gap']:+.3f} "
                    "| RC on/off="
                    f"{_fmt(ablation['comm_on']['rc'])}/"
                    f"{_fmt(ablation['comm_off']['rc'])} "
                    "CPR on/off="
                    f"{_fmt(ablation['comm_on']['cpr'])}/"
                    f"{_fmt(ablation['comm_off']['cpr'])}"
                )

                print("=" * 70)
                print(flush=True)

                writer.add_scalar(
                    "ablation/success_gap",
                    ablation["success_gap"],
                    sampler.level,
                )

                for side in (
                    "comm_on",
                    "comm_off",
                ):
                    for metric in (
                        "rc",
                        "cpr",
                    ):
                        value = ablation[side][
                            metric
                        ]

                        if value is not None:
                            writer.add_scalar(
                                f"ablation/{metric}_{side}",
                                value,
                                sampler.level,
                            )

                history.append(
                    {
                        "level": level_name,
                        "episodes":
                            level_episodes,
                        "elapsed_min": round(
                            (
                                time.time()
                                - start_time
                            )
                            / 60.0,
                            1,
                        ),
                        "buckets":
                            curriculum.status(),
                        "ablation": {
                            "comm_on":
                                ablation[
                                    "comm_on"
                                ],
                            "comm_off":
                                ablation[
                                    "comm_off"
                                ],
                        },
                    }
                )

                curriculum.reset()

                level_episodes = 0

                length_window.clear()
                return_window.clear()
                help_window.clear()
                help_visible_window.clear()
                help_hidden_window.clear()

                if not sampler.advance():

                    _safe_save(
                        ppo,
                        f"{prefix}_final.pt",
                    )

                    stop_reason = "completed"

                else:
                    print(
                        f"Moving to "
                        f"{sampler.name}",
                        flush=True,
                    )

                    if (
                        stop_after_level is not None
                        and mastered_level
                        >= stop_after_level
                    ):
                        stop_reason = (
                            f"stopped after "
                            f"{level_name}"
                        )

                save_latest()

                write_status()

            elif (
                level_episodes
                >= config
                .max_episodes_per_level
            ):
                _safe_save(
                    ppo,
                    f"{prefix}_"
                    f"{sampler.name}"
                    f"_stalled.pt"
                )

                ablation = run_ablation(
                    ppo,
                    sampler,
                    config,
                    tag="stalled",
                )

                print(
                    "  stalled ablation: "
                    "comm_on="
                    f"{ablation['comm_on']['success_rate']:.3f} "
                    "comm_off="
                    f"{ablation['comm_off']['success_rate']:.3f} "
                    "| RC on/off="
                    f"{_fmt(ablation['comm_on']['rc'])}/"
                    f"{_fmt(ablation['comm_off']['rc'])} "
                    "CPR on/off="
                    f"{_fmt(ablation['comm_on']['cpr'])}/"
                    f"{_fmt(ablation['comm_off']['cpr'])}",
                    flush=True,
                )

                stop_reason = (
                    f"stalled at "
                    f"{sampler.name}"
                )

                print(
                    f"STOP: {stop_reason} "
                    f"({level_episodes} "
                    f"episodes without "
                    f"mastery)",
                    flush=True,
                )

                save_latest()

                write_status()

            if stop_reason is not None:
                break

            start_episode(env_index)

        flush_resets()

        if (
            config.recurrent
            and stop_reason is None
        ):
            # 모든 env가 이번 step을 마쳤을 때만 한 step으로
            # 기록한다 (중간에 멈춘 step은 버린다).
            buffer.add_step(
                observations=flat_batch,
                hidden_in=hidden_in,
                episode_starts=
                    episode_starts,
                actions=actions,
                comm_actions=comm_actions,
                log_probs=log_probs,
                values=values,
                rewards=step_rewards,
                dones=step_dones,
            )

            hidden = new_hidden

            episode_starts = step_dones

        rollout_step += 1

        # ====================================================
        # PPO update
        # ====================================================

        if (
            rollout_step
            >= config.rollout_steps
            or stop_reason is not None
        ):
            rollout_step = 0

            if (
                config.recurrent
                and stop_reason is not None
            ):
                # 멈추는 시점의 마지막 조각은 학습하지 않는다
                # (통과/정체 체크포인트는 이미 저장됐다).
                buffer.reset()

            if pending:

                pending_keys = list(
                    pending
                )

                next_values = (
                    ppo.value_batch(
                        np.stack(
                            [
                                states[env_index][
                                    "flat"
                                ][
                                    states[
                                        env_index
                                    ]["agents"].index(
                                        agent
                                    )
                                ]
                                for (
                                    env_index,
                                    agent,
                                ) in pending_keys
                            ]
                        )
                    )
                )

                for key, next_value in zip(
                    pending_keys,
                    next_values,
                ):
                    transition = (
                        pending.pop(key)
                    )

                    transition[
                        "next_value"
                    ] = float(next_value)

                    buffer.add(
                        **transition
                    )

            if len(buffer) > 0:

                # 레벨별 학습 설정 (예: 새 기술을 발견해야 하는
                # 레벨에서만 entropy를 올림). sampler가
                # train_settings()를 제공할 때만 적용한다.
                if hasattr(
                    sampler,
                    "train_settings",
                ):
                    ppo.entropy_coef = (
                        sampler
                        .train_settings()
                        .get(
                            "entropy_coef",
                            config.entropy_coef,
                        )
                    )

                if config.recurrent:

                    _, current_flats = (
                        collect_flats()
                    )

                    metrics = (
                        ppo.update_recurrent(
                            buffer,
                            last_values=(
                                ppo.value_batch_recurrent(
                                    current_flats,
                                    hidden,
                                    episode_starts,
                                )
                            ),
                            gamma=config.gamma,
                            gae_lambda=
                                config.gae_lambda,
                            update_policy=(
                                warmup_remaining
                                <= 0
                            ),
                        )
                    )

                else:
                    metrics = ppo.update(
                        buffer,
                        gamma=config.gamma,
                        gae_lambda=
                            config.gae_lambda,
                        update_policy=(
                            warmup_remaining
                            <= 0
                        ),
                    )

                if warmup_remaining > 0:

                    warmup_remaining -= 1

                    print(
                        "critic warmup: "
                        f"value_loss="
                        f"{metrics['value_loss']:.4f} "
                        f"({warmup_remaining} left)",
                        flush=True,
                    )

                buffer.reset()

                metrics["entropy_coef"] = (
                    ppo.entropy_coef
                )

                last_metrics = metrics

                for name, value in (
                    metrics.items()
                ):
                    writer.add_scalar(
                        f"loss/{name}",
                        value,
                        global_step,
                    )

                save_latest()

    writer.close()

    pool.close()

    probe_env.close()

    print(
        f"{config.run_name} finished: "
        f"{stop_reason}",
        flush=True,
    )

    return stop_reason
