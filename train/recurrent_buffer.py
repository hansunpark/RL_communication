import numpy as np
import torch


# ============================================================
# RecurrentRolloutBuffer
#
# RecurrentPPO용 rollout 버퍼. MultiAgentRolloutBuffer가 transition을
# 에이전트 키별로 하나씩 받는 것과 달리, 매 step 모든 (env, agent)
# 행 N개를 한꺼번에 받아 [T, N] 배열로 쌓는다. comm_runner에서는
# 모든 env가 매 step 에이전트 4명을 다 움직이므로 (episode가 끝나면
# 같은 루프에서 바로 reset) 행 순서가 rollout 내내 고정된다.
#
# 각 step의 hidden_in은 "그 step 직전의 hidden, episode_starts로
# 0으로 만들기 전" 값이다. RecurrentActorCritic.step/unroll이 같은
# 순서로 마스크를 적용하므로, 저장한 hidden_in[t0]에서 다시 펼치면
# rollout 때와 같은 출력이 나온다 (시퀀스 조각 BPTT의 시작점).
# ============================================================


class RecurrentRolloutBuffer:

    def __init__(
        self,
        device,
    ):
        self.device = device

        self.reset()

    def reset(self):

        self.observations = []
        self.hidden_in = []
        self.episode_starts = []
        self.actions = []
        self.comm_actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def __len__(self):

        if not self.actions:
            return 0

        return (
            len(self.actions)
            * len(self.actions[0])
        )

    @property
    def num_steps(self):

        return len(self.actions)

    def add_step(
        self,
        observations,
        hidden_in,
        episode_starts,
        actions,
        comm_actions,
        log_probs,
        values,
        rewards,
        dones,
    ):
        self.observations.append(
            np.asarray(
                observations,
                dtype=np.float32,
            )
        )

        self.hidden_in.append(
            hidden_in.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        for name, value, dtype in (
            ("episode_starts", episode_starts, np.float32),
            ("actions", actions, np.int64),
            ("comm_actions", comm_actions, np.int64),
            ("log_probs", log_probs, np.float32),
            ("values", values, np.float32),
            ("rewards", rewards, np.float32),
            ("dones", dones, np.float32),
        ):
            getattr(self, name).append(
                np.asarray(
                    value,
                    dtype=dtype,
                )
            )

    def compute(
        self,
        last_values,
        gamma=0.99,
        gae_lambda=0.95,
    ):
        """
        GAE를 계산해서 [T, N, ...] 텐서 dict를 반환한다.

        last_values: rollout이 끝난 시점 관측의 value [N].
        dones[t]=1이면 그 step에서 episode가 끝났으므로 다음
        value와 GAE 사슬을 끊는다 (MultiAgentRolloutBuffer와
        같은 규칙: timeout도 종료로 취급).
        """

        values = np.stack(self.values)
        rewards = np.stack(self.rewards)
        dones = np.stack(self.dones)

        steps = values.shape[0]

        advantages = np.zeros_like(values)

        last_gae = np.zeros_like(values[0])

        for t in reversed(range(steps)):

            next_value = (
                np.asarray(
                    last_values,
                    dtype=np.float32,
                )
                if t == steps - 1
                else values[t + 1]
            )

            non_terminal = 1.0 - dones[t]

            delta = (
                rewards[t]
                + gamma
                * next_value
                * non_terminal
                - values[t]
            )

            last_gae = (
                delta
                + gamma
                * gae_lambda
                * non_terminal
                * last_gae
            )

            advantages[t] = last_gae

        returns = advantages + values

        def tensor(array, dtype=torch.float32):

            return torch.as_tensor(
                array,
                dtype=dtype,
                device=self.device,
            )

        return {
            "observations": tensor(
                np.stack(self.observations)
            ),
            "hidden_in": tensor(
                np.stack(self.hidden_in)
            ),
            "episode_starts": tensor(
                np.stack(self.episode_starts)
            ),
            "actions": tensor(
                np.stack(self.actions),
                torch.long,
            ),
            "comm_actions": tensor(
                np.stack(self.comm_actions),
                torch.long,
            ),
            "log_probs": tensor(
                np.stack(self.log_probs)
            ),
            "advantages": tensor(advantages),
            "returns": tensor(returns),
        }
