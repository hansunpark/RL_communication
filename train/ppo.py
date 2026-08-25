import numpy as np

import torch
import torch.nn as nn

from torch.optim import Adam

from .network import ActorCritic


class PPO:

    def __init__(
        self,
        obs_dim,
        num_actions=5,
        learning_rate=3e-4,
        hidden_dim=256,
        clip_coef=0.2,
        value_coef=0.5,
        entropy_coef=0.02,
        max_grad_norm=0.5,
        update_epochs=4,
        minibatch_size=256,
        device="cpu",
    ):

        self.device = torch.device(
            device
        )

        self.network = (
            ActorCritic(
                obs_dim,
                num_actions,
                hidden_dim,
            ).to(
                self.device
            )
        )

        self.optimizer = Adam(
            self.network.parameters(),
            lr=learning_rate,
            eps=1e-5,
        )

        self.clip_coef = clip_coef

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

        _, value = (
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
            old_log_probs,
            advantages,
            returns,
        ) = buffer.compute(
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

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
                        + self.minibatch_size
                    ]
                )

                (
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
                    )
                )

                ratio = torch.exp(
                    new_log_prob
                    - old_log_probs[
                        batch_idx
                    ]
                )

                batch_adv = (
                    advantages[
                        batch_idx
                    ]
                )

                loss_1 = (
                    -batch_adv
                    * ratio
                )

                loss_2 = (
                    -batch_adv
                    * torch.clamp(
                        ratio,
                        1
                        - self.clip_coef,
                        1
                        + self.clip_coef,
                    )
                )

                policy_loss = (
                    torch.max(
                        loss_1,
                        loss_2,
                    ).mean()
                )

                value_loss = (
                    0.5
                    * (
                        new_value
                        - returns[
                            batch_idx
                        ]
                    )
                    .pow(2)
                    .mean()
                )

                entropy_mean = (
                    entropy.mean()
                )

                loss = (
                    policy_loss
                    + self.value_coef
                    * value_loss
                    - self.entropy_coef
                    * entropy_mean
                )

                self.optimizer.zero_grad()

                loss.backward()

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
                np.mean(
                    policy_losses
                ),

            "value_loss":
                np.mean(
                    value_losses
                ),

            "entropy":
                np.mean(
                    entropies
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
    ):

        checkpoint = (
            torch.load(
                path,
                map_location=
                    self.device,
            )
        )

        self.network.load_state_dict(
            checkpoint[
                "network"
            ]
        )