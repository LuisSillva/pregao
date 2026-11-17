import numpy as np
import pandas as pd
import pytest

from pregao import comparison as cmp


def long_df(series: dict[str, pd.Series]) -> pd.DataFrame:
    frames = [pd.DataFrame({"date": s.index, "ticker": t, "adj_close": s.values}) for t, s in series.items()]
    return pd.concat(frames, ignore_index=True)


def test_wide_prices_mantem_ordem_e_datas_diferentes():
    a = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    b = pd.Series([10.0, 30.0], index=pd.to_datetime(["2024-01-02", "2024-01-04"]))  # feriado no dia 3
    wide = cmp.wide_prices(long_df({"A": a, "B": b}), ["B", "A"])
    assert list(wide.columns) == ["B", "A"]
    assert np.isnan(wide.loc["2024-01-03", "B"])


def test_normalized_comeca_em_100():
    wide = pd.DataFrame({"A": [50.0, 75.0], "B": [np.nan, 8.0]}, index=pd.bdate_range("2024-01-01", periods=2))
    n = cmp.normalized(wide)
    assert n["A"].tolist() == [100.0, 150.0]
    assert n["B"].dropna().iloc[0] == 100.0


def test_beta_recupera_inclinacao_conhecida():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=800)
    market = rng.normal(0, 0.01, len(idx))
    noise = rng.normal(0, 0.005, len(idx))
    index_prices = pd.Series(100 * np.exp(np.cumsum(market)), index=idx)
    asset_prices = pd.Series(50 * np.exp(np.cumsum(1.5 * market + noise)), index=idx)
    b = cmp.beta(asset_prices, index_prices, "^BVSP")
    assert b.beta == pytest.approx(1.5, abs=0.05)
    assert 0.8 < b.r2 < 1
    assert len(b.points) == len(idx) - 1


def test_beta_usa_so_dias_em_comum():
    idx = pd.bdate_range("2024-01-01", periods=10)
    index_prices = pd.Series(np.linspace(100, 110, 10), index=idx)
    asset_prices = pd.Series(np.linspace(10, 12, 10), index=idx).drop(idx[5])
    assert len(cmp.beta(asset_prices, index_prices, "^BVSP").points) < 9


def test_correlacao_semanal_e_simetrica():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2020-01-01", periods=300)
    base = np.cumsum(rng.normal(0, 0.01, 300))
    wide = pd.DataFrame({"A": np.exp(base), "B": np.exp(base + rng.normal(0, 0.001, 300))}, index=idx)
    c = cmp.weekly_correlation(wide)
    assert c.loc["A", "B"] == pytest.approx(c.loc["B", "A"])
    assert c.loc["A", "B"] > 0.9
    assert c.loc["A", "A"] == pytest.approx(1.0)


def test_risk_return_por_ativo():
    idx = pd.bdate_range("2020-01-01", periods=600)
    wide = pd.DataFrame({"A": np.linspace(100, 200, 600), "B": np.linspace(100, 50, 600)}, index=idx)
    rr = cmp.risk_return(wide).set_index("ticker")
    assert rr.loc["A", "cagr"] > 0 > rr.loc["B", "cagr"]


def test_cores_estaveis_ao_remover():
    first = cmp.assign_colors(["A", "B", "C"], None, 8)
    assert first == {"A": 0, "B": 1, "C": 2}
    after = cmp.assign_colors(["B", "C"], first, 8)
    assert after == {"B": 1, "C": 2}  # B e C não mudam de cor


def test_cores_novo_ativo_pega_slot_livre():
    current = {"B": 1, "C": 2}
    assert cmp.assign_colors(["B", "C", "D"], current, 8)["D"] == 0


def test_benchmark():
    assert cmp.benchmark_for("PETR4.SA") == "^BVSP"
    assert cmp.benchmark_for("AAPL") == "^GSPC"
