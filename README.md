# Pregão

Painel para visualizar ações da B3 e dos EUA e testar modelos de previsão de forma honesta: cada modelo é comparado com uma baseline ingênua num backtest walk-forward, e o erro aparece na tela, inclusive quando o modelo perde.

> Em construção. O projeto está sendo feito por partes; cada parte tem seus commits.

## Como os dados chegam

```
pregao/tickers.py ──► GitHub Actions (seg a sex, depois do fechamento da B3)
                         │  scripts/update_data.py baixa o histórico no Yahoo Finance
                         │  se um ticker falhar, mantém os dados antigos dele
                         ▼
                      data/prices.parquet ──► app Dash (só lê o arquivo)
```

O app nunca chama o Yahoo Finance diretamente. O yfinance não é uma API oficial e sofre com rate limit, então os dados são baixados uma vez por dia e ficam versionados no repositório.

## Rodando localmente

Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python scripts/update_data.py    # baixa os dados
pytest
```
