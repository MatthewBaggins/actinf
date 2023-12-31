from typing import Literal

import torch as t

from src.constants import ENV_TEMP_MU, ENV_TEMP_SIGMA


def make_envs(
    n_envs: int,
    *,
    temp_mu: float = ENV_TEMP_MU,  # temperature mean
    temp_sigma: float = ENV_TEMP_SIGMA,  # temperature standard deviation
    # power usage should always start at zero for both thermostats, so I guess irrelevant?
    # power1_lambda: float,
    # power2_lambda: float,
) -> t.Tensor:  # [n_envs 3 (temp, power1, power2)]
    temp = t.randn(n_envs) * temp_sigma + temp_mu
    power1 = t.zeros(n_envs)
    power2 = t.zeros(n_envs)
    return t.stack((temp, power1, power2), dim=1)


def observe_envs(envs: t.Tensor, thermostat_i: Literal[1, 2]) -> t.Tensor:
    return envs.index_select(1, t.tensor([0, thermostat_i]))


# How much one binary action nudges temperature
ACTION_SIZE = 1e-3  # TODO better name


def act_in_envs(
    envs: t.Tensor,  # [n_envs 3]
    action_scores: t.Tensor,  # [n_envs n_actions(2 for now)]
) -> t.Tensor:
    actions = action_scores[:, 1] - action_scores[:, 0]
    new_envs = envs.clone().detach()
    new_envs[:, 0] += ACTION_SIZE * actions
    return new_envs
