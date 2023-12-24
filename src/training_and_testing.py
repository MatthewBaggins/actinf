from typing import NamedTuple

import torch as t
from tqdm import tqdm

from src.constants import DEFAULT_N_ENVS, DEFAULT_N_ROUNDS
from src.envs import act_in_envs, make_envs
from src.thermostat import Thermostat


class TrainingHistory(NamedTuple):
    gains: list[float]
    n_rounds: int
    n_envs: int


def train(
    model: Thermostat,
    optimizer: t.optim.Optimizer,
    n_rounds: int = DEFAULT_N_ENVS,
    n_envs: int = DEFAULT_N_ROUNDS,
    *,
    progressbar: bool = True,
) -> TrainingHistory:
    gains: list[float] = []

    for round_i in tqdm(range(n_rounds), disable=not progressbar):
        # Reinitialize envs
        envs = make_envs(n_envs)

        # Compute action scores on observation
        action_scores: t.Tensor = model(envs[:, :1])

        # Take action, transforming environment
        new_envs = act_in_envs(envs, action_scores)

        # Compute prior and posterior preference scores
        pref_pre = model.temp_densities(envs[:, 0])
        pref_post = model.temp_densities(new_envs[:, 0])

        # Compute gain from the difference between post and pre preferences
        pref_diff = pref_post - pref_pre
        gain = pref_diff.pow(2).mean()

        # Backprop
        gain.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Append to history
        gain = gain.item()
        gains.append(gain)

        # if round_i % 100 == 0:
        #     print(f"[{round_i}] {gain=}")

    return TrainingHistory(gains, n_rounds, n_envs)


class TestingHistory(NamedTuple):
    prefs: list[float]
    env_history: t.Tensor  # [n_rounds n_envs 3]
    n_rounds: int


def test(
    model: Thermostat, envs: t.Tensor, n_rounds: int = 100, *, progressbar: bool = True
) -> TestingHistory:
    with t.no_grad():
        env_history = t.empty(n_rounds + 1, *envs.shape)
        env_history[0] = envs
        prefs: list[float] = [model.temp_densities(envs[:, 0]).mean().item()]
        for round_i in tqdm(range(1, n_rounds + 1), disable=not progressbar):
            # Compute action scores on observation
            action_scores: t.Tensor = model(envs[:, :1])

            # Take action, transforming environment
            envs = act_in_envs(envs, action_scores)
            env_history[round_i] = envs

            # Compute preference scores
            pref = model.temp_densities(envs[:, 0]).mean().item()
            prefs.append(pref)

    return TestingHistory(prefs, env_history, n_rounds)
