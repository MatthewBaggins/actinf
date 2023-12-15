import torch as t
from torch import nn
from scipy.stats import norm, expon


class Thermostat(nn.Module):
    def __init__(self, temp_mu: float = 20, hidden_dim: int = 3) -> None:
        super().__init__()
        self.temp = norm(temp_mu)
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2)
        )

    def forward(
        self, obs: t.Tensor  # [2] temp pow1 (fully states are fully observable)
    ) -> t.Tensor:  # [2] logits of action (-1 or 1)
        x = obs
        for layer in self.net:
            x = layer(x)
        return x
