import numpy as np
import pandas as pd
import pytest

from pregao import forecast


def test_verdict():
    assert forecast.verdict(0.1, 0.01)[1] == "good"
    assert forecast.verdict(-0.1, 0.01)[1] == "bad"
    assert forecast.verdict(0.1, 0.4)[1] == "neutral"
    assert forecast.verdict(0.0, np.nan)[0] == "referência"


def test_cumulative_edge_sobe_quando_modelo_ganha():
    dates = pd.bdate_range("2024-01-01", periods=4)
    y = [0.1, -0.1, 0.2, 0.0]
    rows = []
    for d, yt in zip(dates, y):
        rows.append({"date": d, "model": "Baseline (zero)", "y_true": yt, "y_pred": 0.0})
        rows.append({"date": d, "model": "Média histórica", "y_true": yt, "y_pred": 0.0})
        rows.append({"date": d, "model": "Regressão linear", "y_true": yt, "y_pred": yt})  # perfeito
        rows.append({"date": d, "model": "LightGBM", "y_true": yt, "y_pred": -yt})  # sempre do lado errado
    edge = forecast.cumulative_edge(pd.DataFrame(rows), "retorno")
    assert edge["Média histórica"].eq(0).all()
    assert edge["Regressão linear"].iloc[-1] == pytest.approx(0.4)
    assert edge["LightGBM"].iloc[-1] == pytest.approx(-0.4)
    assert edge["Regressão linear"].is_monotonic_increasing


def test_overview_so_desafiantes():
    m = pd.DataFrame({
        "ticker": ["A", "A"], "target": ["retorno"] * 2, "model": ["Baseline (zero)", "LightGBM"],
        "is_baseline": [True, False], "skill": [0.0, -0.1], "p_value": [np.nan, 0.01],
    })
    o = forecast.overview(m, "retorno")
    assert o["model"].tolist() == ["LightGBM"]
    assert bool(o["significant"].iloc[0])
