"""Roda o backtest de todos os modelos, em todos os ativos, e salva os resultados.

Chamado pelo `scripts/train_models.py` no GitHub Actions, uma vez por semana.
O site só lê os arquivos gerados aqui, como faz com os preços.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pregao import comparison, data
from pregao.ml import backtest, features, metrics, models
from pregao.tickers import ASSETS

ML_DIR = data.ROOT / "data" / "ml"
INTERVAL = (0.10, 0.90)  # intervalo de 80%
RESIDUAL_WINDOW = 504  # erros dos últimos ~2 anos para montar o intervalo

STOCKS = [a.ticker for a in ASSETS if a.ticker not in comparison.INDEXES]


@dataclass
class Results:
    metrics: list[dict] = field(default_factory=list)
    backtest: list[pd.DataFrame] = field(default_factory=list)
    latest: list[dict] = field(default_factory=list)
    importance: list[pd.DataFrame] = field(default_factory=list)


def evaluate(y_true: pd.Series, preds: dict[str, pd.Series], target: str, horizon: int) -> list[dict]:
    """Uma linha de métricas por modelo, sempre comparando com a baseline."""
    base_name = models.baseline_name(target)
    base = preds[base_name]
    base_mae = metrics.mae(y_true, base)
    rows = []
    for name, pred in preds.items():
        mae = metrics.mae(y_true, pred)
        dm, p = (np.nan, np.nan) if name == base_name else metrics.diebold_mariano(y_true, pred, base, horizon)
        row = {
            "model": name,
            "is_baseline": name == base_name,
            "mae": mae,
            "rmse": metrics.rmse(y_true, pred),
            "skill": metrics.skill(mae, base_mae),
            "dm": dm,
            "p_value": p,
            "n": len(y_true),
            "start": y_true.index.min(),
            "end": y_true.index.max(),
        }
        if target == "retorno":
            row["direction"] = metrics.direction_accuracy(y_true, pred)
            row["up_rate"] = metrics.up_rate(y_true)
        rows.append(row)
    return rows


def run_ticker(ticker: str, prices: pd.DataFrame, horizon: int, out: Results) -> None:
    history = data.series(prices, ticker)
    index = data.series(prices, comparison.benchmark_for(ticker))["adj_close"]
    X, Y = features.build(history, index, horizon)
    as_of = X.index[-1]

    for target in features.TARGETS:
        y = Y[target]
        preds = {name: backtest.walk_forward(X, y, make, horizon) for name, make in models.MODELS[target].items()}
        common = preds[models.baseline_name(target)].index
        y_true = y.loc[common]

        for row in evaluate(y_true, preds, target, horizon):
            out.metrics.append({"ticker": ticker, "target": target, **row})

        for name, make in models.MODELS[target].items():
            pred = preds[name]
            # Previsão de hoje, com intervalo montado a partir dos erros recentes do backtest:
            # se nos últimos 2 anos o modelo errou entre -3% e +4% em 80% das vezes,
            # o intervalo de hoje é previsão - 3% até previsão + 4%.
            model, today = backtest.fit_latest(X, y, make)
            resid = (y_true - pred).iloc[-RESIDUAL_WINDOW:]
            low, high = today + resid.quantile(INTERVAL[0]), today + resid.quantile(INTERVAL[1])
            if target == "volatilidade":
                low = max(low, 0.0)
            out.latest.append({
                "ticker": ticker, "target": target, "model": name, "as_of": as_of,
                "prediction": today, "low": low, "high": high,
            })
            imp = models.feature_importance(model, list(X.columns))
            if imp is not None:
                out.importance.append(pd.DataFrame({
                    "ticker": ticker, "target": target, "feature": imp.index, "gain": imp.values,
                }))

            # Guardamos só uma previsão a cada `horizon` dias: os alvos de dias vizinhos
            # se sobrepõem (4 dos 5 dias são os mesmos), então pontos diários repetiriam
            # quase a mesma informação e inchariam o arquivo 5 vezes.
            sample = pred.iloc[::horizon]
            out.backtest.append(pd.DataFrame({
                "date": sample.index, "ticker": ticker, "target": target, "model": name,
                "y_true": y_true.loc[sample.index].to_numpy(np.float32),
                "y_pred": sample.to_numpy(np.float32),
            }))


def run(prices: pd.DataFrame, tickers: list[str] = STOCKS, horizon: int = features.HORIZON,
        log=print) -> Results:
    out = Results()
    for i, ticker in enumerate(tickers, 1):
        log(f"[{i}/{len(tickers)}] {ticker}")
        run_ticker(ticker, prices, horizon, out)
    return out


def save(res: Results, horizon: int, directory: Path = ML_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res.metrics).to_parquet(directory / "metrics.parquet", index=False)
    pd.concat(res.backtest, ignore_index=True).to_parquet(directory / "backtest.parquet", index=False)
    pd.DataFrame(res.latest).to_parquet(directory / "latest.parquet", index=False)
    pd.concat(res.importance, ignore_index=True).to_parquet(directory / "importance.parquet", index=False)
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "horizon": horizon,
        "min_train": backtest.MIN_TRAIN,
        "retrain_every": backtest.STEP,
        "interval": INTERVAL,
        "features": features.FEATURES,
    }
    (directory / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def load(directory: Path = ML_DIR) -> dict[str, pd.DataFrame] | None:
    if not (directory / "metrics.parquet").exists():
        return None
    out = {name: pd.read_parquet(directory / f"{name}.parquet")
           for name in ("metrics", "backtest", "latest", "importance")}
    out["meta"] = json.loads((directory / "meta.json").read_text())
    return out
