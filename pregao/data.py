"""Camada de dados: baixa preços do Yahoo Finance e mantém um cache em parquet.

O app nunca chama o Yahoo diretamente. Um job agendado roda `scripts/update_data.py`
uma vez por dia, que usa as funções daqui para gerar `data/prices.parquet`.
O app só lê esse arquivo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "prices.parquet"

# Formato "longo": uma linha por (data, ticker). É mais fácil de filtrar, juntar e
# salvar do que o formato "largo" (uma coluna por ticker), e não desperdiça espaço
# quando os mercados têm feriados diferentes (B3 e EUA não abrem nos mesmos dias).
COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]

# Se a nova coleta de um ticker vier com menos da metade das linhas que já temos,
# provavelmente o Yahoo devolveu dados incompletos. Nesse caso mantemos os antigos.
MIN_ROWS_RATIO = 0.5

Fetcher = Callable[[str, str], pd.DataFrame]


def yahoo_fetcher(ticker: str, start: str) -> pd.DataFrame:
    """Baixa o histórico diário de um ticker no Yahoo Finance."""
    import yfinance as yf  # import aqui para os testes não dependerem da rede

    return yf.download(
        ticker,
        start=start,
        auto_adjust=False,  # queremos o preço bruto E o ajustado (veja normalize)
        actions=False,
        progress=False,
        threads=False,
        multi_level_index=False,
    )


def normalize(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Converte a tabela do yfinance para o nosso formato padrão.

    - `close` é o preço que apareceu na tela da bolsa naquele dia.
    - `adj_close` é o preço corrigido por dividendos e desdobramentos. É ele que
      usamos para calcular retornos: sem o ajuste, um desdobramento 1:2 apareceria
      como uma queda de 50% que nunca aconteceu.
    """
    if raw is None or raw.empty:
        return pd.DataFrame(columns=COLUMNS)

    df = raw.copy()
    # Versões do yfinance às vezes devolvem colunas em dois níveis (campo, ticker).
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()  # só a data, sem horário

    df = df.rename_axis("date").reset_index()
    df["ticker"] = ticker
    df = df[COLUMNS].dropna(subset=["close", "adj_close"])
    df = df.drop_duplicates(subset="date", keep="last").sort_values("date")
    df["volume"] = df["volume"].fillna(0).astype("int64")
    return df.reset_index(drop=True)


@dataclass
class UpdateReport:
    updated: dict[str, int] = field(default_factory=dict)  # ticker -> linhas
    kept_old: dict[str, str] = field(default_factory=dict)  # ticker -> motivo
    failed: dict[str, str] = field(default_factory=dict)  # sem dado novo nem antigo

    @property
    def ok(self) -> bool:
        return bool(self.updated)


def update(
    tickers: list[str],
    existing: pd.DataFrame | None,
    fetcher: Fetcher = yahoo_fetcher,
    start: str = "2015-01-01",
) -> tuple[pd.DataFrame, UpdateReport]:
    """Baixa o histórico completo de cada ticker e junta com o cache existente.

    Por que baixar tudo de novo em vez de só os dias novos? Porque, quando uma
    empresa paga dividendo, o Yahoo recalcula o `adj_close` de TODO o histórico
    anterior. Anexar só os dias novos deixaria a série inconsistente.
    São poucos ativos, então baixar tudo é barato.

    Se um ticker falhar, mantemos os dados antigos dele: uma fonte instável não
    pode apagar o que já temos.
    """
    existing = existing if existing is not None else pd.DataFrame(columns=COLUMNS)
    report = UpdateReport()
    frames: list[pd.DataFrame] = []

    for ticker in tickers:
        old = existing[existing["ticker"] == ticker]
        try:
            new = normalize(fetcher(ticker, start), ticker)
            problem = None
            if new.empty:
                problem = "coleta vazia"
            elif len(new) < MIN_ROWS_RATIO * len(old):
                problem = f"coleta incompleta ({len(new)} linhas, antes {len(old)})"
        except Exception as exc:  # rede, rate limit, mudança na API...
            new, problem = None, f"erro: {exc}"

        if problem is None:
            frames.append(new)
            report.updated[ticker] = len(new)
        elif not old.empty:
            frames.append(old)
            report.kept_old[ticker] = problem
        else:
            report.failed[ticker] = problem

    if not frames:
        return pd.DataFrame(columns=COLUMNS), report
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["ticker", "date"]).reset_index(drop=True), report


def save(df: pd.DataFrame, path: Path = DATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_parquet(path)


def series(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Histórico de um ticker, indexado por data."""
    return df[df["ticker"] == ticker].set_index("date").drop(columns="ticker")
