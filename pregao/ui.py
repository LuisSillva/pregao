"""Peças de interface usadas por mais de uma página.

Páginas não devem importar umas às outras: o Dash carrega cada arquivo de pages/
por conta própria, e importar uma página de outra faz ela ser registrada duas vezes.
O que for compartilhado mora aqui.
"""

from pregao.tickers import BY_TICKER

GRAPH_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]}


def ticker_label(ticker: str) -> str:
    asset = BY_TICKER[ticker]
    return f"{ticker.removesuffix('.SA')} · {asset.name}"
