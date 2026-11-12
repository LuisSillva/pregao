"""Lista curada de ativos que o app acompanha.

Ficar numa lista fixa (em vez de aceitar qualquer ticker digitado) é proposital:
os dados são baixados uma vez por dia por um job agendado e ficam em cache.
Assim o app nunca depende do Yahoo Finance estar no ar na hora em que alguém abre.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Asset:
    ticker: str  # código no Yahoo Finance (ações da B3 terminam em .SA)
    name: str
    market: str  # "B3" ou "EUA"
    sector: str
    currency: str


ASSETS: list[Asset] = [
    # Índices de referência, usados para comparar o desempenho das ações
    Asset("^BVSP", "Ibovespa", "B3", "Índice", "BRL"),
    Asset("^GSPC", "S&P 500", "EUA", "Índice", "USD"),
    # B3
    Asset("PETR4.SA", "Petrobras PN", "B3", "Petróleo e gás", "BRL"),
    Asset("VALE3.SA", "Vale ON", "B3", "Mineração", "BRL"),
    Asset("ITUB4.SA", "Itaú Unibanco PN", "B3", "Bancos", "BRL"),
    Asset("BBDC4.SA", "Bradesco PN", "B3", "Bancos", "BRL"),
    Asset("BBAS3.SA", "Banco do Brasil ON", "B3", "Bancos", "BRL"),
    Asset("B3SA3.SA", "B3 ON", "B3", "Serviços financeiros", "BRL"),
    Asset("WEGE3.SA", "WEG ON", "B3", "Bens industriais", "BRL"),
    Asset("ABEV3.SA", "Ambev ON", "B3", "Bebidas", "BRL"),
    Asset("SUZB3.SA", "Suzano ON", "B3", "Papel e celulose", "BRL"),
    Asset("PRIO3.SA", "PRIO ON", "B3", "Petróleo e gás", "BRL"),
    Asset("RENT3.SA", "Localiza ON", "B3", "Locação de veículos", "BRL"),
    Asset("MGLU3.SA", "Magazine Luiza ON", "B3", "Varejo", "BRL"),
    # EUA
    Asset("AAPL", "Apple", "EUA", "Tecnologia", "USD"),
    Asset("MSFT", "Microsoft", "EUA", "Tecnologia", "USD"),
    Asset("NVDA", "NVIDIA", "EUA", "Semicondutores", "USD"),
    Asset("GOOGL", "Alphabet", "EUA", "Tecnologia", "USD"),
    Asset("AMZN", "Amazon", "EUA", "Varejo e nuvem", "USD"),
]

BY_TICKER: dict[str, Asset] = {a.ticker: a for a in ASSETS}
