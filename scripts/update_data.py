"""Atualiza data/prices.parquet. Roda todo dia no GitHub Actions.

Uso: python scripts/update_data.py [--start 2015-01-01]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pregao import data  # noqa: E402
from pregao.tickers import ASSETS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2015-01-01")
    args = parser.parse_args()

    existing = data.load()
    df, report = data.update([a.ticker for a in ASSETS], existing, start=args.start)

    for ticker, rows in report.updated.items():
        print(f"ok        {ticker:10} {rows} linhas")
    for ticker, reason in report.kept_old.items():
        print(f"mantido   {ticker:10} {reason}")
    for ticker, reason in report.failed.items():
        print(f"falhou    {ticker:10} {reason}")

    if not report.ok:
        print("Nenhum ticker atualizado. O arquivo não foi alterado.")
        return 1

    data.save(df)
    print(f"Salvo em {data.DATA_PATH.relative_to(data.ROOT)} ({len(df)} linhas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
