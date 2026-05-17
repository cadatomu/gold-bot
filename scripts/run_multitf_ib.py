"""Backtest multi-timeframe (15m, H1, H4) con datos históricos de IB — GC CONTFUT.

NO abre órdenes. Solo descarga datos históricos de IB Gateway y corre el backtest.

Comisiones IB reales para GC (mini COMEX gold futures, 100 oz/contrato):
  - Comisión: ~$2.05/lado × 2 = $4.10 RT por contrato
  - Spread:   1 tick = $0.10/oz (punto mínimo GC)
  - Slippage: 1 tick en órdenes de mercado

Uso:
  python scripts/run_multitf_ib.py              # backtest con defaults
  python scripts/run_multitf_ib.py --optimize   # optimiza Calmar (0.5% riesgo fijo)
  python scripts/run_multitf_ib.py --aggressive # optimiza retorno mensual (riesgo 1-8%)
  python scripts/run_multitf_ib.py --trials 400
  python scripts/run_multitf_ib.py --no-cache   # re-descarga datos aunque existan
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pickle
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for line in (ROOT / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from src.backtest.runner import BrokerConfig
from src.backtest.runner_advanced import run_backtest_advanced
from src.strategy.adaptive import AdaptiveParams, AdaptiveScalp15m
from src.strategy.regime import RegimeParams
from src.strategy.scalp_15m import build_h1_trend

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config GC Futures (IB / COMEX) ────────────────────────────────────────────
#
# GC: oro COMEX, 100 oz/contrato, tick mínimo $0.10/oz
#   pip_value_per_lot = 100 oz × $0.10/tick = $10 por tick por contrato
#   commission_per_lot_rt = IB tiered: ~$2.05/lado × 2 lados = $4.10 RT
#   spread_pips = 1 tick ($0.10/oz → 1 pip con point_size=0.10)
#   slippage_pips = 1 tick en órdenes de mercado

BROKER = BrokerConfig(
    initial_equity        = 10_000.0,
    spread_pips           = 1.0,
    commission_per_lot_rt = 4.10,
    slippage_pips         = 1.0,
    risk_per_trade_pct    = 0.005,
    pip_value_per_lot     = 10.0,
    point_size            = 0.10,    # tick mínimo GC = $0.10/oz
)

MONTHS = 20
DAYS   = int(MONTHS * 30.5)   # ~610 días
CACHE_DIR = ROOT / "data" / "ib_cache"


# ── IB helpers ────────────────────────────────────────────────────────────────

def _ib_connect():
    from ib_insync import IB, Contract, util
    util.logToConsole(level=40)   # solo errores
    ib = IB()
    host = os.environ.get("IB_HOST", "127.0.0.1")
    port = int(os.environ.get("IB_PORT", "4002"))
    ib.connect(host, port, clientId=10, readonly=True)

    contract = Contract(symbol="GC", secType="CONTFUT",
                        exchange="COMEX", currency="USD")
    ib.qualifyContracts(contract)
    print(f"  [IB] Conectado {host}:{port} — contrato calificado: {contract.localSymbol}")
    return ib, contract


def _bars_to_df(bars) -> pd.DataFrame:
    rows = [{"time":   b.date, "open":  b.open,
             "high":   b.high, "low":   b.low,
             "close":  b.close, "volume": b.volume}
            for b in bars if b.open > 0]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # ib_insync devuelve datetime (intraday) o date (daily) con formatDate=2.
    # Normalizamos a UTC en ambos casos.
    def _to_utc(x):
        ts = pd.Timestamp(x)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    df["time"] = df["time"].apply(_to_utc)
    df = df.set_index("time").sort_index()
    return df[~df.index.duplicated(keep="last")]


# Pausa entre requests para respetar pacing limit de IB (60 req/10 min)
_SLEEP = 5.0

# Contratos GC explícitos para paginar 15m más allá de 60 días.
# Cada tupla: (year, month_num, approx_last_trade_date_str)
# last_trade_date = 3er-to-last business day del mes de vencimiento
_GC_ROLL_CONTRACTS = [
    (2024,  8, "20240827"),   # GCQ4
    (2024, 10, "20241029"),   # GCV4
    (2024, 12, "20241227"),   # GCZ4
    (2025,  2, "20250226"),   # GCG5
    (2025,  4, "20250428"),   # GCJ5
    (2025,  6, "20250626"),   # GCM5
    (2025,  8, "20250827"),   # GCQ5
    (2025, 10, "20251029"),   # GCV5
    (2025, 12, "20251229"),   # GCZ5
    (2026,  2, "20260226"),   # GCG6
    (2026,  4, "20260428"),   # GCJ6
    # GCM6 (Jun 2026) — contrato actual, usar endDateTime=''
]


def _download_contfut(ib, contract, bar_size: str) -> pd.DataFrame:
    """
    Descarga datos CONTFUT con endDateTime='' (único valor permitido para CONTFUT).
    IB error 10339: CONTFUT no acepta endDateTime con fechas pasadas.
    Solución: pedir '2 Y' en un solo request — cubre los 20 meses necesarios.
    Probado: H4='2 Y' devuelve 3585 barras, Daily='2 Y' devuelve 508 barras.
    """
    bars = ib.reqHistoricalData(
        contract,
        endDateTime    = "",     # CONTFUT solo acepta '' (más reciente)
        durationStr    = "2 Y",  # ~730 días calendario > 610 días (20 meses)
        barSizeSetting = bar_size,
        whatToShow     = "TRADES",
        useRTH         = False,
        formatDate     = 2,
        keepUpToDate   = False,
    )
    time.sleep(_SLEEP)
    return _bars_to_df(bars) if bars else pd.DataFrame()


def _download_by_fut_segments(ib, bar_size: str, total_days: int) -> pd.DataFrame:
    """
    Descarga datos intraday paginando hacia atrás con contratos FUT explícitos.

    CONTFUT no permite endDateTime < now (IB error 10339), y tiene un límite de
    barras por request (~3000 para H1, impidiendo cubrir 20 meses).
    Solución: usar contratos FUT por mes de vencimiento y concatenar.

    Probado: GCG5 (expirado Feb 2025) con includeExpired + endDateTime → devuelve datos.
    Aplica a 15m y H1 (H4 y Daily sí funcionan con CONTFUT '2 Y').
    """
    from ib_insync import Contract as IBContract

    frames    = []
    today     = datetime.now(timezone.utc)
    target_start = today - timedelta(days=total_days)

    # Segmento actual: contrato front-month activo (GCM6) con endDateTime=''
    cur_fut = IBContract(symbol="GC", secType="FUT",
                         exchange="COMEX", currency="USD",
                         lastTradeDateOrContractMonth="202606")
    ib.qualifyContracts(cur_fut)
    bars = ib.reqHistoricalData(
        cur_fut, endDateTime="", durationStr="75 D",
        barSizeSetting=bar_size, whatToShow="TRADES",
        useRTH=False, formatDate=2, keepUpToDate=False,
    )
    if bars:
        df_seg = _bars_to_df(bars)
        frames.append(df_seg)
        print(f"    GCM6 {bar_size}: {len(bars)} barras  "
              f"{df_seg.index[0].date()} → {df_seg.index[-1].date()}")
    time.sleep(_SLEEP)

    # Segmentos históricos: FUT por mes de vencimiento, hacia atrás
    for year, month, end_date_str in reversed(_GC_ROLL_CONTRACTS):
        end_dt = datetime.strptime(end_date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
        if end_dt < target_start:
            break   # ya cubrimos el período completo

        fut = IBContract(symbol="GC", secType="FUT",
                         exchange="COMEX", currency="USD",
                         lastTradeDateOrContractMonth=f"{year}{month:02d}",
                         includeExpired=True)
        details = ib.reqContractDetails(fut)
        if details:
            fut = details[0].contract
            fut.includeExpired = True
        time.sleep(1)

        end_str = end_dt.strftime("%Y%m%d %H:%M:%S UTC")
        bars = ib.reqHistoricalData(
            fut, endDateTime=end_str, durationStr="75 D",
            barSizeSetting=bar_size, whatToShow="TRADES",
            useRTH=False, formatDate=2, keepUpToDate=False,
        )
        if bars:
            df_seg = _bars_to_df(bars)
            frames.append(df_seg)
            print(f"    GC{year}{month:02d} {bar_size}: {len(bars)} barras  "
                  f"{df_seg.index[0].date()} → {df_seg.index[-1].date()}")
        else:
            print(f"    GC{year}{month:02d} {bar_size}: sin datos")
        time.sleep(_SLEEP)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    # Recortar al período objetivo
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=total_days)
    return df[df.index >= cutoff]


def _download_tf(ib, contract, bar_size: str, total_days: int) -> pd.DataFrame:
    """
    Estrategia de descarga según timeframe:
    - 15m y H1: FUT explícitos por contrato (CONTFUT no pagina / tiene límite de barras)
    - H4 y Daily: CONTFUT '2 Y' en un solo request (confirmado: H4=3585 barras, Daily=508)
    """
    if bar_size in ("15 mins", "1 hour"):
        return _download_by_fut_segments(ib, bar_size, total_days)
    return _download_contfut(ib, contract, bar_size)


# ── Descarga / caché ──────────────────────────────────────────────────────────

def download_all(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"gc_{MONTHS}m.pkl"

    if use_cache and cache_file.exists():
        age_h = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_h < 12:
            print(f"  Usando caché ({age_h:.1f}h de antigüedad): {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        else:
            print(f"  Caché expirada ({age_h:.1f}h), re-descargando…")

    print("  Conectando a IB Gateway…")
    ib, contract = _ib_connect()

    timeframe_configs = [
        ("15m",   "15 mins", DAYS),
        ("H1",    "1 hour",  DAYS),
        ("H4",    "4 hours", DAYS),
        ("Daily", "1 day",   DAYS + 200),   # extra para el filtro de tendencia
    ]

    dfs: dict[str, pd.DataFrame] = {}
    for label, bar_size, total_days in timeframe_configs:
        print(f"  Descargando {label} ({total_days} días)…")
        df = _download_tf(ib, contract, bar_size, total_days)
        if df.empty:
            print(f"    ADVERTENCIA: sin datos para {label}")
        else:
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=total_days)
            df = df[df.index >= cutoff]
            print(f"    {len(df):,} barras  {df.index[0].date()} → {df.index[-1].date()}")
        dfs[label] = df

    ib.disconnect()
    print("  [IB] Desconectado.")

    with open(cache_file, "wb") as f:
        pickle.dump(dfs, f)
    print(f"  Datos guardados en caché: {cache_file}")

    return dfs


# ── Build dataframe con filtro de tendencia ───────────────────────────────────

def build_df(entry_df: pd.DataFrame, trend_df: pd.DataFrame) -> pd.DataFrame:
    h1 = build_h1_trend(trend_df)
    h1.index = h1.index.tz_localize("UTC") if h1.index.tzinfo is None else h1.index
    df = entry_df.copy()
    df["h1_trend"] = h1.reindex(entry_df.index, method="ffill").fillna(0).astype(int)
    return df


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_broker_config():
    print("\nConfiguración de comisiones IB (GC futures):")
    print(f"  point_size            : ${BROKER.point_size:.2f}/oz (tick mínimo GC)")
    print(f"  pip_value_per_lot     : ${BROKER.pip_value_per_lot:.2f} por tick/contrato")
    print(f"  commission_per_lot_rt : ${BROKER.commission_per_lot_rt:.2f} RT por contrato")
    print(f"  spread_pips           : {BROKER.spread_pips:.1f} tick(s)")
    print(f"  slippage_pips         : {BROKER.slippage_pips:.1f} tick(s)")
    print(f"  risk_per_trade        : {BROKER.risk_per_trade_pct*100:.1f}% del capital")


def _print_summary(result, title: str):
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")
    print(f"  Total return  : {result.total_return_pct:+.2f}%")
    print(f"  Sharpe        : {result.sharpe:.3f}")
    print(f"  Calmar        : {result.calmar:.3f}")
    print(f"  Max DD        : {result.max_drawdown_pct:.2f}%")
    print(f"  Win rate      : {result.win_rate_pct:.1f}%")
    print(f"  Total trades  : {result.total_trades}")
    print(f"  Profit factor : {result.profit_factor:.2f}")


def _print_monthly(trades_df: pd.DataFrame):
    if trades_df.empty:
        print("  (sin trades)")
        return
    td = trades_df[trades_df["exit_reason"] != "PARTIAL"].copy()
    td["month"] = pd.to_datetime(td["exit_time"]).dt.to_period("M")
    m = td.groupby("month").agg(
        trades  = ("pnl_usd", "count"),
        pnl_usd = ("pnl_usd", "sum"),
        win_pct = ("pnl_usd", lambda x: (x > 0).mean() * 100),
        avg     = ("pnl_usd", "mean"),
        mejor   = ("pnl_usd", "max"),
        peor    = ("pnl_usd", "min"),
    ).reset_index()
    m["month"] = m["month"].astype(str)
    print(f"\n{'─'*90}")
    print(f"  {'Mes':<10} {'Trades':>6} {'PnL USD':>10} {'Win%':>7} {'Avg':>9} {'Mejor':>10} {'Peor':>10}")
    print(f"{'─'*90}")
    for _, r in m.iterrows():
        mark = " ★" if r["pnl_usd"] > 0 else "  "
        print(f"  {r['month']:<10} {r['trades']:>6} {r['pnl_usd']:>10.2f}"
              f" {r['win_pct']:>7.1f} {r['avg']:>9.2f}"
              f" {r['mejor']:>10.2f} {r['peor']:>10.2f}{mark}")
    print(f"{'─'*90}")
    profitable = (m["pnl_usd"] > 0).sum()
    print(f"  {'TOTAL':<10} {m['trades'].sum():>6} {m['pnl_usd'].sum():>10.2f}"
          f"   Meses positivos: {profitable}/{len(m)}")


def _print_direction(trades_df: pd.DataFrame):
    if trades_df.empty or "direction" not in trades_df.columns:
        return
    td = trades_df[trades_df["exit_reason"] != "PARTIAL"]
    if td.empty:
        return
    print("\nPor dirección:")
    grp = td.groupby("direction").agg(
        trades   = ("pnl_usd", "count"),
        pnl_usd  = ("pnl_usd", "sum"),
        win_rate = ("pnl_usd", lambda x: f"{(x > 0).mean()*100:.1f}%"),
        avg      = ("pnl_usd", "mean"),
    )
    print(grp.round(2).to_string())


# ── Optimización ──────────────────────────────────────────────────────────────

def _build_params(b: dict) -> AdaptiveParams:
    return AdaptiveParams(
        regime = RegimeParams(
            adx_trend_min   = b["regime_adx"],
            slope_threshold = b["regime_slope"],
        ),
        adx_min          = b["adx_min"],
        rsi_min          = b["rsi_min"],
        rsi_max          = b["rsi_max"],
        sl_atr_mult      = b["sl_atr_mult"],
        tp_atr_mult      = b["tp_atr_mult"],
        trail_start_mult = b["trail_start"],
        trail_dist_mult  = b["trail_dist"],
        partial_close_r  = b["partial_close"],
        use_session_filter = b.get("session_filter", True),
    )


def _suggest_strategy_params(trial: optuna.Trial) -> AdaptiveParams:
    return _build_params({
        "regime_adx":    trial.suggest_float("regime_adx",   20.0, 40.0, step=1.0),
        "regime_slope":  trial.suggest_float("regime_slope",  0.01, 0.20, step=0.01),
        "adx_min":       trial.suggest_float("adx_min",       18.0, 40.0, step=1.0),
        "rsi_min":       trial.suggest_float("rsi_min",       25.0, 50.0, step=5.0),
        "rsi_max":       trial.suggest_float("rsi_max",       50.0, 75.0, step=5.0),
        "sl_atr_mult":   trial.suggest_float("sl_atr_mult",    0.8,  3.0, step=0.1),
        "tp_atr_mult":   trial.suggest_float("tp_atr_mult",    1.5,  6.0, step=0.1),
        "trail_start":   trial.suggest_float("trail_start",    1.5,  4.0, step=0.1),
        "trail_dist":    trial.suggest_float("trail_dist",     0.2,  1.5, step=0.1),
        "partial_close": trial.suggest_float("partial_close",  0.5,  2.5, step=0.25),
        "session_filter": trial.suggest_categorical("session_filter", [True, False]),
    })


def _objective(trial: optuna.Trial, df: pd.DataFrame) -> float:
    params = _suggest_strategy_params(trial)
    result = run_backtest_advanced(df, AdaptiveScalp15m(params), BROKER)
    if result.total_trades < 8:
        return float("-inf")
    if result.max_drawdown_pct < -15.0:   # max_drawdown_pct es negativo (ej: -5.2%)
        return float("-inf")
    score = result.calmar
    return float(score) if score > -10 else float("-inf")


def _objective_aggressive(trial: optuna.Trial, df: pd.DataFrame, n_months: float) -> float:
    # Riesgo por trade como parámetro: 1-8%.
    # Justificación: para 6%/mes necesitamos ~9.4× el retorno base (0.64%/mes a 0.5% risk).
    # 9.4 × 0.5% = 4.7% risk. Para 8%/mes → 12.5× → 6.25% risk.
    # DD escala linealmente con risk: 0.70% base × (risk_pct/0.005) → tope 30%.
    risk_pct = trial.suggest_float("risk_pct", 0.01, 0.08, step=0.005)
    broker   = dataclasses.replace(BROKER, risk_per_trade_pct=risk_pct)

    params = _suggest_strategy_params(trial)
    result = run_backtest_advanced(df, AdaptiveScalp15m(params), broker)
    if result.total_trades < 15:
        return float("-inf")
    if result.max_drawdown_pct < -30.0:
        return float("-inf")
    monthly_ret = result.total_return_pct / n_months
    return float(monthly_ret) if monthly_ret > 0 else float("-inf")


def optimise(df: pd.DataFrame, label: str, n_trials: int,
             aggressive: bool = False) -> tuple[AdaptiveParams, float]:
    """Devuelve (params, risk_pct_usado)."""
    n_months = (df.index[-1] - df.index[0]).days / 30.5

    if aggressive:
        print(f"\n  Optimizando AGRESIVO {label} — {n_trials} trials")
        print(f"  Objetivo: retorno mensual máximo | DD máx 30% | risk 1-8% | min 15 trades")
        obj = lambda t: _objective_aggressive(t, df, n_months)
    else:
        print(f"\n  Optimizando {label} — {n_trials} trials (Calmar | DD máx 15% | 0.5% risk)")
        obj = lambda t: _objective(t, df)

    study = optuna.create_study(
        direction = "maximize",
        sampler   = optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)

    b     = study.best_params
    score = study.best_value

    if aggressive:
        risk_pct = b["risk_pct"]
        print(f"\n  Mejor retorno mensual {label}: {score:.2f}%/mes  "
              f"(risk={risk_pct*100:.1f}%/trade)")
    else:
        risk_pct = BROKER.risk_per_trade_pct
        print(f"\n  Mejor Calmar {label}: {score:.3f}")

    for k, v in b.items():
        if k == "risk_pct":
            print(f"    {k}: {v*100:.2f}%")
        elif isinstance(v, float):
            print(f"    {k}: {v:.3f}")
        else:
            print(f"    {k}: {v}")

    return _build_params(b), risk_pct


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize",   action="store_true")
    parser.add_argument("--aggressive", action="store_true",
                        help="Optimiza retorno mensual (risk 1-8%%, DD máx 30%%)")
    parser.add_argument("--trials",     type=int,  default=200)
    parser.add_argument("--no-cache",   action="store_true", dest="no_cache")
    args = parser.parse_args()

    run_optimize   = args.optimize or args.aggressive
    run_aggressive = args.aggressive

    _print_broker_config()

    print(f"\nDescargando {MONTHS} meses de datos GC (IB Gateway)…")
    dfs = download_all(use_cache=not args.no_cache)

    for label, df in dfs.items():
        if df.empty:
            print(f"ERROR: sin datos para {label}. Verifica que IB Gateway esté activo.")
            sys.exit(1)

    timeframes = [
        ("15m", dfs["15m"],   dfs["H1"],    False),
        ("H1",  dfs["H1"],    dfs["H4"],    False),
        ("H4",  dfs["H4"],    dfs["Daily"], True),
    ]

    for label, entry_df, trend_df, disable_session in timeframes:
        print(f"\n{'#'*64}")
        print(f"  TIMEFRAME: {label} — GC CONTFUT COMEX (IB)")
        print(f"{'#'*64}")

        df = build_df(entry_df, trend_df)

        if run_optimize:
            params, risk_pct = optimise(df, label, args.trials, aggressive=run_aggressive)
            broker = dataclasses.replace(BROKER, risk_per_trade_pct=risk_pct)
        else:
            params   = AdaptiveParams(use_session_filter=not disable_session)
            broker   = BROKER
            risk_pct = BROKER.risk_per_trade_pct

        strategy = AdaptiveScalp15m(params)
        result   = run_backtest_advanced(df, strategy, broker)

        n_months = (df.index[-1] - df.index[0]).days / 30.5
        monthly  = result.total_return_pct / n_months

        title = f"AdaptiveScalp {label} — IB/GC ({MONTHS}m)"
        if run_aggressive:
            title += f"  [risk={risk_pct*100:.1f}%]"
        _print_summary(result, title)
        print(f"  Retorno mensual  : {monthly:+.2f}%/mes  ({n_months:.1f} meses)")
        _print_monthly(result.trades)
        if not result.trades.empty:
            print("\nSalidas:")
            print(result.trades["exit_reason"].value_counts().to_string())
            _print_direction(result.trades)


if __name__ == "__main__":
    main()
