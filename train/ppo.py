import numpy as np

import torch
import torch.nn as nn

from torch.optim import Adam

from .network import (
    ActorCritic,
)


class PPO:

    # 러너가 hidden state를 다뤄야 하는지 판단할 때 쓴다
    # (RecurrentPPO는 True).
    recurrent = False

    def _build_network(
        self,
        obs_dim,
        num_actions,
        hidden_dim,
    ):
        # 하위 클래스(RecurrentPPO)가 다른 네트워크를 쓰도록
        # 바꿔 끼우는 지점.
        return ActorCritic(
            obs_dim=obs_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
        )

    def __init__(
        self,
        obs_dim,
        num_actions=5,
        learning_rate=3e-4,
        hidden_dim=256,
        clip_coef=0.2,
        value_coef=0.5,
        entropy_coef=0.03,
        max_grad_norm=0.5,
        update_epochs=4,
        minibatch_size=256,
        device="cpu",
        comm_entropy_coef=None,
        target_kl=None,
    ):

        self.device = (
            torch.device(
                device
            )
        )

        self.network = (
            self._build_network(
                obs_dim=obs_dim,
                num_actions=num_actions,
                hidden_dim=hidden_dim,
            ).to(
                self.device
            )
        )

        self.optimizer = (
            Adam(
                self.network.parameters(),

                lr=learning_rate,

                eps=1e-5,
            )
        )

        self.clip_coef = (
            clip_coef
        )

        self.value_coef = (
            value_coef
        )

        self.entropy_coef = (
            entropy_coef
        )

        # 통신 head 전용 entropy 계수. None이면 entropy_coef를
        # 그대로 써서 기존(두 head entropy 합에 같은 계수)과
        # 동일하게 동작한다.
        #
        # 분리한 이유: Stage 8B1에서 이동 entropy_coef=0.03이
        # 실패 위주 rollout의 약한 신호를 압도해 이동 entropy가
        # 0.7 -> 1.4(최대 ln5=1.61)로 치솟고 성공률이 40% -> 4%로
        # 무너졌다. 이동 쪽은 계수를 낮춰야 하지만, 통신 head는
        # 아직 쓸모를 배우기 전에 HELP 0%/100%로 굳지 않도록
        # entropy를 따로 유지해야 한다.
        self.comm_entropy_coef = (
            entropy_coef
            if comm_entropy_coef is None
            else comm_entropy_coef
        )

        # update 도중 approx KL이 target_kl의 1.5배를 넘으면 남은
        # minibatch/epoch를 건너뛴다 (None이면 기존과 동일하게 항상
        # update_epochs를 다 돈다).
        #
        # 도입 이유: comm_scratch(lr 3e-4)가 S3, S3a, S4에서 반복해서
        # 이미 하던 과제까지 잃으며 무너졌다 (S4: S3b 과제 85% -> 18%).
        # 같은 기간 lr 1e-4인 stage8_comm은 10개 넘는 레벨 동안 한 번도
        # 무너지지 않았다. update 한 번이 정책을 너무 멀리 옮기지 못하게
        # 막는 안전장치다.
        self.target_kl = target_kl

        self.max_grad_norm = (
            max_grad_norm
        )

        self.update_epochs = (
            update_epochs
        )

        self.minibatch_size = (
            minibatch_size
        )

    @torch.no_grad()
    def act(
        self,
        observation,
    ):

        obs = torch.tensor(
            observation,

            dtype=torch.float32,

            device=self.device,
        ).unsqueeze(0)

        (
            action,
            comm_action,
            log_prob,
            _,
            value,
        ) = (
            self.network
            .get_action_and_value(
                obs
            )
        )

        return (
            int(
                action.item()
            ),

            int(
                comm_action.item()
            ),

            float(
                log_prob.item()
            ),

            float(
                value.item()
            ),
        )

    @torch.no_grad()
    def act_batch(
        self,
        observations,
    ):
        """
        act()의 batch 버전. observations는 (N, obs_dim)
        배열이고, (actions, comm_actions, log_probs, values)
        numpy 배열 4개를 반환한다. 여러 env x 여러 agent를
        forward 한 번으로 처리하기 위함.
        """

        obs = torch.as_tensor(
            observations,

            dtype=torch.float32,

            device=self.device,
        )

        (
            action,
            comm_action,
            log_prob,
            _,
            value,
        ) = (
            self.network
            .get_action_and_value(
                obs
            )
        )

        return (
            action.cpu().numpy(),
            comm_action.cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().numpy(),
        )

    @torch.no_grad()
    def value_batch(
        self,
        observations,
    ):

        obs = torch.as_tensor(
            observations,

            dtype=torch.float32,

            device=self.device,
        )

        _, _, value = (
            self.network(obs)
        )

        return value.cpu().numpy()

    @torch.no_grad()
    def value(
        self,
        observation,
    ):

        obs = torch.tensor(
            observation,

            dtype=torch.float32,

            device=self.device,
        ).unsqueeze(0)

        _, _, value = (
            self.network(obs)
        )

        return float(
            value.item()
        )

    def update(
        self,
        buffer,
        gamma,
        gae_lambda,
        update_policy=True,
    ):
        """
        update_policy=False면 critic head(마지막 선형층)만
        학습한다 (encoder/actor/comm_actor는 그대로).

        warm-start 직후 critic warmup용이다. 이전 stage의
        critic은 "거의 항상 성공"에 맞춰져 있어서, 새 stage에서
        실패가 많아지면 초반 update에서 잘한 행동에도 음의
        advantage가 붙고 기존 정책이 망가진다 (stage8_comm
        8B1에서 성공률 24% -> 5%). encoder는 actor와 공유하므로
        value loss만 걸어도 정책이 흔들려서, feature를
        detach하고 critic head만 맞춘다.
        """

        if not update_policy:
            return self._update_critic_head(
                buffer,
                gamma,
                gae_lambda,
            )

        (
            observations,
            actions,
            comm_actions,
            old_log_probs,
            advantages,
            returns,
        ) = (
            buffer.compute(
                gamma=gamma,

                gae_lambda=
                    gae_lambda,
            )
        )

        if (
            advantages.numel()
            > 1
        ):

            advantages = (
                advantages
                - advantages.mean()
            ) / (
                advantages.std()
                + 1e-8
            )

        batch_size = (
            observations.shape[0]
        )

        indices = np.arange(
            batch_size
        )

        policy_losses = []
        value_losses = []
        entropies = []

        move_entropies = []

        comm_entropies = []

        approx_kls = []

        stop_early = False

        for _ in range(
            self.update_epochs
        ):

            if stop_early:
                break

            np.random.shuffle(
                indices
            )

            for start in range(
                0,
                batch_size,
                self.minibatch_size,
            ):

                batch_idx = (
                    indices[
                        start:
                        start
                        +
                        self.minibatch_size
                    ]
                )

                (
                    new_log_prob,
                    move_entropy,
                    comm_entropy,
                    new_value,
                ) = (
                    self.network
                    .evaluate_actions(
                        observations[
                            batch_idx
                        ],

                        actions[
                            batch_idx
                        ],

                        comm_actions[
                            batch_idx
                        ],
                    )
                )

                log_ratio = (
                    new_log_prob
                    -
                    old_log_probs[
                        batch_idx
                    ]
                )

                ratio = torch.exp(
                    log_ratio
                )

                with torch.no_grad():
                    approx_kl = (
                        (ratio - 1.0)
                        - log_ratio
                    ).mean().item()

                approx_kls.append(approx_kl)

                if (
                    self.target_kl is not None
                    and approx_kl
                    > 1.5 * self.target_kl
                ):
                    stop_early = True

                    break

                batch_adv = (
                    advantages[
                        batch_idx
                    ]
                )

                unclipped_loss = (
                    -batch_adv
                    * ratio
                )

                clipped_loss = (
                    -batch_adv
                    *
                    torch.clamp(
                        ratio,

                        1.0
                        - self.clip_coef,

                        1.0
                        + self.clip_coef,
                    )
                )

                policy_loss = (
                    torch.max(
                        unclipped_loss,
                        clipped_loss,
                    ).mean()
                )

                value_loss = (
                    0.5
                    *
                    (
                        new_value
                        -
                        returns[
                            batch_idx
                        ]
                    )
                    .pow(2)
                    .mean()
                )

                move_entropy_mean = (
                    move_entropy.mean()
                )

                comm_entropy_mean = (
                    comm_entropy.mean()
                )

                entropy_mean = (
                    move_entropy_mean
                    + comm_entropy_mean
                )

                total_loss = (
                    policy_loss
                    +
                    self.value_coef
                    * value_loss
                    -
                    self.entropy_coef
                    * move_entropy_mean
                    -
                    self.comm_entropy_coef
                    * comm_entropy_mean
                )

                self.optimizer.zero_grad()

                total_loss.backward()

                nn.utils.clip_grad_norm_(
                    self.network.parameters(),

                    self.max_grad_norm,
                )

                self.optimizer.step()

                policy_losses.append(
                    policy_loss.item()
                )

                value_losses.append(
                    value_loss.item()
                )

                entropies.append(
                    entropy_mean.item()
                )

                move_entropies.append(
                    move_entropy_mean.item()
                )

                comm_entropies.append(
                    comm_entropy_mean.item()
                )

        return {
            "policy_loss":
                float(
                    np.mean(
                        policy_losses
                    )
                )
                if policy_losses
                else 0.0,

            "value_loss":
                float(
                    np.mean(
                        value_losses
                    )
                )
                if value_losses
                else 0.0,

            "entropy":
                float(
                    np.mean(
                        entropies
                    )
                )
                if entropies
                else 0.0,

            "move_entropy":
                float(
                    np.mean(
                        move_entropies
                    )
                )
                if move_entropies
                else 0.0,

            "approx_kl":
                float(
                    np.mean(approx_kls)
                ),

            "kl_early_stop":
                float(stop_early),

            "comm_entropy":
                float(
                    np.mean(
                        comm_entropies
                    )
                )
                if comm_entropies
                else 0.0,
        }

    def _update_critic_head(
        self,
        buffer,
        gamma,
        gae_lambda,
    ):

        (
            observations,
            _,
            _,
            _,
            _,
            returns,
        ) = buffer.compute(
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        with torch.no_grad():
            features = (
                self.network.encoder(
                    observations
                )
            )

        indices = np.arange(
            observations.shape[0]
        )

        value_losses = []

        for _ in range(
            self.update_epochs
        ):

            np.random.shuffle(indices)

            for start in range(
                0,
                len(indices),
                self.minibatch_size,
            ):
                batch_idx = indices[
                    start:
                    start
                    + self.minibatch_size
                ]

                value = (
                    self.network.critic(
                        features[batch_idx]
                    )
                    .squeeze(-1)
                )

                value_loss = (
                    0.5
                    * (
                        value
                        - returns[batch_idx]
                    )
                    .pow(2)
                    .mean()
                )

                # set_to_none(기본값)라서 critic 외 파라미터는
                # grad가 None이고 Adam이 건너뛴다.
                self.optimizer.zero_grad()

                value_loss.backward()

                self.optimizer.step()

                value_losses.append(
                    value_loss.item()
                )

        return {
            "policy_loss": 0.0,
            "value_loss": float(
                np.mean(value_losses)
            ),
        }

    def save(
        self,
        path,
    ):

        torch.save(
            {
                "network":
                    self.network
                    .state_dict(),

                "optimizer":
                    self.optimizer
                    .state_dict(),
            },

            path,
        )

    def load(
        self,
        path,
        load_optimizer=True,
    ):

        checkpoint = (
            torch.load(
                path,

                map_location=
                    self.device,
            )
        )

        # strict=False: comm_actor head가 없는 구 체크포인트
        # (통신 도입 이전, 예: stage_8A_size_mastered.pt)를
        # warm-start로 불러올 때 이동 관련 가중치(encoder/
        # actor/critic)만 옮겨오고 comm_actor는 새로 초기화된
        # 상태로 둔다.
        load_result = (
            self.network.load_state_dict(
                checkpoint[
                    "network"
                ],

                strict=False,
            )
        )

        if (
            load_result.missing_keys
            or
            load_result.unexpected_keys
        ):

            print(
                "PPO.load(): checkpoint "
                "architecture differs from "
                "current network."
            )

            if load_result.missing_keys:

                print(
                    "  missing (freshly "
                    "initialized): "
                    f"{load_result.missing_keys}"
                )

            if (
                load_result
                .unexpected_keys
            ):

                print(
                    "  unexpected "
                    "(ignored): "
                    f"{load_result.unexpected_keys}"
                )

        if (
            load_optimizer
            and
            "optimizer"
            in checkpoint
        ):

            try:
                self.optimizer.load_state_dict(
                    checkpoint[
                        "optimizer"
                    ]
                )

            except (
                ValueError,
                RuntimeError,
            ) as error:

                print(
                    "PPO.load(): optimizer "
                    "state incompatible "
                    "with current network "
                    "(architecture changed), "
                    f"skipping: {error}"
                )