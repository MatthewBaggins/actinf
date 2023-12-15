import torch as t

# TODO write tests for pdf normal


def pdf_normal(x: t.Tensor, *, mu: float, sigma: float) -> t.Tensor:
    return t.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * (2 * t.pi) ** 0.5)
