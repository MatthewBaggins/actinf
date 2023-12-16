import torch as t
from torch import nn
from torch.distributions import Categorical, Normal

# from src.stats import pdf_normal

HIDDEN_DIM = 3


class Thermostat(nn.Module):
    def __init__(self, *, temp_mu: float = 20, temp_sigma: float = 1) -> None:
        super().__init__()
        self.temp_mu = temp_mu
        self.temp_sigma = temp_sigma
        self.temp = Normal(temp_mu, temp_sigma)

        self.hidden_dim = HIDDEN_DIM
        self.net = nn.Sequential(
            nn.Linear(2, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, 2)
        )

    def forward(
        self,
        obs: t.Tensor,  # [2] temp pow1 (fully states are fully observable)
    ) -> t.Tensor:  # [2] logits of action (-1 or 1)
        x = obs
        for layer in self.net:
            x = layer(x)
        return x

    def loss_fn(
        self,
        obs: t.Tensor,  # [2] (temp pow1)
    ) -> t.Tensor:
        temp = obs[:, 0]
        densities = self.temp_densities(temp)
        loss = -densities.pow(2).mean()
        return loss

    def temp_densities(self, temp: t.Tensor) -> t.Tensor:
        return self.temp.log_prob(temp).exp()

    def sample_action(
        self, action_probs: t.Tensor  # [n_envs n_actions]
    ) -> t.Tensor:  # [n_envs 1] (row-wise: unit tensor [index of chosen action])
        return Categorical(action_probs).sample()
