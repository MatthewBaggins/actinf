import torch as t

from scipy.stats import norm
from src.stats import pdf_normal


def test_pdf_normal():
    x = t.rand(1000) * 100  # 100 floats from 0 to 100
    mu = 1
    sigma = 2
    scipy_densities = t.tensor(norm.pdf(x, mu, sigma), dtype=t.float32)
    our_densities = pdf_normal(x, mu=mu, sigma=sigma)
    assert t.allclose(scipy_densities, our_densities)


def test_pdf_normal_fail():
    x = t.rand(1000) * 100  # 100 floats from 0 to 100
    mu = 1
    sigma = 2
    # swapped mu and sigma in norm.pdf
    scipy_densities = t.tensor(norm.pdf(x, sigma, mu), dtype=t.float32)
    our_densities = pdf_normal(x, mu=mu, sigma=sigma)
    assert not t.allclose(scipy_densities, our_densities)
