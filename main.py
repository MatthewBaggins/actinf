import torch as t

from src.envs import act_in_envs, make_envs, observe_envs
from torch.distributions.categorical import Categorical

from src.utils import seed
from src.thermostat import Thermostat


SEED = 42
N_ENVS = 10
N_ROUNDS = 100


def loss_fn(pref_pre: t.Tensor, pref_post: t.Tensor) -> t.Tensor:
    post_minus_pre = -(pref_post - pref_pre)
    return post_minus_pre.pow(2).mean()


def main() -> None:
    seed(SEED)
    envs = make_envs(N_ENVS)
    # 3 = D_ENV (temp, pow1, pow2)
    envs_history: t.Tensor = t.empty(N_ROUNDS, N_ENVS, 3)
    loss_history: list[float] = []

    model = Thermostat()
    lr = 1e-4
    optimizer = t.optim.AdamW(model.parameters(), lr)
    print(envs.shape)
    for round_i in range(N_ROUNDS):
        envs_history[round_i] = envs

        # Compute action scores on observation
        action_scores: t.Tensor = model(envs[:, :1])

        # Take action, transforming environment
        new_envs = act_in_envs(envs, action_scores)

        # Compute prior and posterior preference scores
        pref_pre = model.temp_densities(envs[:, 0])
        pref_post = model.temp_densities(new_envs[:, 0])

        # Compute loss from prior and posterior preference scores
        loss = loss_fn(pref_pre, pref_post)

        # Backprop
        loss.backward()
        optimizer.step()

        loss = loss.item()
        loss_history.append(loss)

        envs = new_envs.detach()
        mean_curr_temp = envs[:, 0].mean()

        optimizer.zero_grad()

        print(f"[{round_i}] {loss=:.5f}; {mean_curr_temp=:.5f}")
        # break


if __name__ == "__main__":
    main()
