import numpy as np

import torch
import torch.nn as nn

from torch.optim import Adam

from .network import (
    ActorCritic,
)


class PPO:

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
    ):

        self.device = (
            torch.device(
                device
            )
        )

        self.network = (
            ActorCritic(
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
    ):

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

        for _ in range(
            self.update_epochs
        ):

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
                    _,
                    _,
                    new_log_prob,
                    entropy,
                    new_value,
                ) = (
                    self.network
                    .get_action_and_value(
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

                ratio = torch.exp(
                    new_log_prob
                    -
                    old_log_probs[
                        batch_idx
                    ]
                )

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

                entropy_mean = (
                    entropy.mean()
                )

                total_loss = (
                    policy_loss
                    +
                    self.value_coef
                    * value_loss
                    -
                    self.entropy_coef
                    * entropy_mean
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

        return {
            "policy_loss":
                float(
                    np.mean(
                        policy_losses
                    )
                ),

            "value_loss":
                float(
                    np.mean(
                        value_losses
                    )
                ),

            "entropy":
                float(
                    np.mean(
                        entropies
                    )
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