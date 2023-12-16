import torch as t

from src.envs import make_envs, observe_envs
from torch.distributions.categorical import Categorical

from src.utils import seed
from src.thermostat import Thermostat


SEED = 42
N_ENVS = 10
N_ROUNDS = 100


def main() -> None:
    seed(SEED)
    envs = make_envs(N_ENVS)
    model = Thermostat()
    obs = observe_envs(envs, 1)
    for round_i in range(N_ROUNDS):
        action_probs = model(obs)
        actions = model.sample_action(action_probs)
        print(round_i, action_probs.shape, actions.shape)
        print(actions)
        break


if __name__ == "__main__":
    main()
