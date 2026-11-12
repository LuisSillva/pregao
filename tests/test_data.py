import pandas as pd
import pytest

from pregao import data


def fake_yahoo(n_days: int, start: str = "2024-01-02", multiindex: bool = False, tz: bool = False):
    """Tabela no formato que o yfinance devolve."""
    idx = pd.bdate_range(start, periods=n_days, tz="America/Sao_Paulo" if tz else None)
    df = pd.DataFrame(
        {
            "Open": 10.0,
            "High": 11.0,
            "Low": 9.0,
            "Close": [10.0 + i for i in range(n_days)],
            "Adj Close": [9.0 + i for i in range(n_days)],
            "Volume": 1000,
        },
        index=idx,
    )
    if multiindex:
        df.columns = pd.MultiIndex.from_product([df.columns, ["PETR4.SA"]])
    return df


@pytest.mark.parametrize("multiindex", [False, True])
@pytest.mark.parametrize("tz", [False, True])
def test_normalize_formato_padrao(multiindex, tz):
    df = data.normalize(fake_yahoo(5, multiindex=multiindex, tz=tz), "PETR4.SA")
    assert list(df.columns) == data.COLUMNS
    assert len(df) == 5
    assert (df["ticker"] == "PETR4.SA").all()
    assert df["date"].dt.tz is None
    assert df["adj_close"].iloc[0] == 9.0


def test_normalize_descarta_dias_sem_preco():
    raw = fake_yahoo(3)
    raw.iloc[1, raw.columns.get_loc("Close")] = None
    assert len(data.normalize(raw, "X")) == 2


def test_normalize_vazio():
    assert data.normalize(pd.DataFrame(), "X").empty


def test_update_baixa_todos():
    df, report = data.update(["A", "B"], None, fetcher=lambda t, s: fake_yahoo(10))
    assert set(report.updated) == {"A", "B"}
    assert len(df) == 20


def test_update_mantem_antigo_quando_fonte_falha():
    existing, _ = data.update(["A", "B"], None, fetcher=lambda t, s: fake_yahoo(10))

    def flaky(ticker, start):
        if ticker == "B":
            raise ConnectionError("429 Too Many Requests")
        return fake_yahoo(11)

    df, report = data.update(["A", "B"], existing, fetcher=flaky)
    assert report.updated == {"A": 11}
    assert "B" in report.kept_old
    assert len(df[df["ticker"] == "B"]) == 10


def test_update_rejeita_coleta_incompleta():
    existing, _ = data.update(["A"], None, fetcher=lambda t, s: fake_yahoo(100))
    df, report = data.update(["A"], existing, fetcher=lambda t, s: fake_yahoo(10))
    assert "A" in report.kept_old
    assert len(df) == 100


def test_update_substitui_historico_ajustado():
    """Depois de um dividendo, o adj_close antigo muda; a versão nova deve prevalecer."""
    existing, _ = data.update(["A"], None, fetcher=lambda t, s: fake_yahoo(10))
    revised = fake_yahoo(10)
    revised["Adj Close"] -= 0.5
    df, _ = data.update(["A"], existing, fetcher=lambda t, s: revised)
    assert df["adj_close"].iloc[0] == 8.5


def test_save_load(tmp_path):
    df, _ = data.update(["A"], None, fetcher=lambda t, s: fake_yahoo(5))
    path = tmp_path / "p.parquet"
    data.save(df, path)
    pd.testing.assert_frame_equal(data.load(path), df)
    assert data.series(df, "A").index.name == "date"
