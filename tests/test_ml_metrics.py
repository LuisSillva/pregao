import numpy as np
import pytest

from pregao.ml import metrics

def test_skill():
    assert metrics.skill(0.9, 1.0) == pytest.approx(0.1)
    assert metrics.skill(1.2, 1.0) == pytest.approx(-0.2)


def test_direction_ignora_previsao_zero():
    assert metrics.direction_accuracy([1, -1, 1], [0.5, 0.5, 0.0]) == pytest.approx(0.5)
    assert np.isnan(metrics.direction_accuracy([1, -1], [0, 0]))


def test_diebold_mariano_detecta_modelo_melhor():
    rng = np.random.default_rng(3)
    y = rng.normal(0, 1, 2000)
    good = y + rng.normal(0, 0.5, 2000)
    bad = y + rng.normal(0, 1.0, 2000)
    stat, p = metrics.diebold_mariano(y, good, bad, horizon=5)
    assert stat < 0 and p < 0.01


def test_diebold_mariano_nao_inventa_diferenca():
    rng = np.random.default_rng(4)
    y = rng.normal(0, 1, 2000)
    a = y + rng.normal(0, 1, 2000)
    b = y + rng.normal(0, 1, 2000)
    _, p = metrics.diebold_mariano(y, a, b, horizon=5)
    assert p > 0.05
