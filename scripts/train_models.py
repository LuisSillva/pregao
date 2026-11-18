"""Treina e avalia os modelos de previsão. Roda toda semana no GitHub Actions.

Uso: python scripts/train_models.py [--tickers PETR4.SA VALE3.SA]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pregao import data  # noqa: E402
from pregao.ml import features, train  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=train.STOCKS)
    args = parser.parse_args()

    prices = data.load()
    if prices.empty:
        print("Sem dados. Rode scripts/update_data.py antes.")
        return 1

    t0 = time.time()
    res = train.run(prices, args.tickers)
    train.save(res, features.HORIZON)
    print(f"Pronto em {time.time() - t0:.0f}s. Resultados em data/ml/")

    m = __import__("pandas").DataFrame(res.metrics)
    summary = m[~m["is_baseline"]].groupby(["target", "model"])["skill"].median().unstack(0)
    print("\nSkill mediano contra a baseline (positivo = melhor):")
    print(summary.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
