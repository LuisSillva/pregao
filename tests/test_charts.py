import pytest

from pregao.charts import dodge


def test_dodge_afasta_rotulos_proximos():
    out = dodge([1.0, 1.01, 5.0], min_gap=0.5)
    assert out[1] - out[0] >= 0.5 - 1e-9
    assert out[0] < out[1] < out[2]  # ordem vertical preservada


def test_dodge_nao_mexe_em_rotulos_distantes():
    assert dodge([0.0, 10.0], min_gap=1.0) == pytest.approx([0.0, 10.0])
