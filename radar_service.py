"""Taiwan Alpha Radar V16 - production scan with market/sector/stock hierarchy."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import os

import numpy as np
import pandas as pd

from market_data import DailyPriceStore, ResearchDataClient, fetch_twse_universe, _taipei_timestamp
from policy_engine import generate_trade_plan, evaluate_entry_state, setup_from_df
from return_first_model import estimate_horizon_return, estimate_all_horizons
from industry_profile import fine_industry, theme_bucket
from trading_calendar import calendar_reference
from operational_tools import atomic_write_json

OPERATIONS_VERSION = "v16.0.0-final-evidence"
MIN_PRODUCTION_UNIVERSE = 300
MIN_PRODUCTION_EVALUATED = 80


@dataclass
class RunSettings:
    reference_size: int = 600
    candidate_size: int = 1000
    research_pool_per_horizon: int = 12
    history_period: str = "5y"
    model_family: str = "full"  # price_only | business_confirmed | flow_confirmed | full
    order_mode: str = "next_open"
    commission: float = 0.001425
    sell_tax: float = 0.003
    slippage: float = 0.0005
    notional: float = 100000.0
    min_ev_short: float = 0.0
    min_ev_mid: float = 0.0
    min_ev_long: float = 0.0
    min_price: float = 10.0
    min_avg_turnover: float = 10_000_000.0


def load_dashboard(path: Path, include_features: bool = False) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def compact_session_dashboard(snap: dict | None) -> dict:
    if not snap or not isinstance(snap, dict):
        return {}
    out = dict(snap)
    out["charts"] = {}
    return out


def compact_doctor_result(dr: dict | None) -> dict:
    return dict(dr) if isinstance(dr, dict) else {}


def remove_saved_dashboard(path: Path) -> bool:
    try:
        if path.exists():
            path.unlink()
        return True
    except Exception:
        return False


def snapshot_bytes(snap: dict) -> bytes:
    return json.dumps(snap, ensure_ascii=False, indent=2).encode("utf-8")


def chart_on_demand(snap: dict | None, ticker: str, data_dir: Path, allow_fetch: bool = False) -> dict | None:
    if not ticker:
        return None
    store = DailyPriceStore(data_dir / "daily_prices.sqlite")
    df = store.get_prices(ticker)
    if df.empty and allow_fetch:
        store.batch_fetch_and_update([ticker], period="1y")
        df = store.get_prices(ticker)
    if df.empty:
        return None
    tail = df.tail(260)
    return {
        "dates": tail.index.strftime("%Y-%m-%d").tolist(),
        "ohlcv": tail[["Open", "High", "Low", "Close", "Volume"]].to_numpy().tolist(),
    }


def _market_context(store: DailyPriceStore) -> dict:
    df = store.get_prices("^TWII")
    if df.empty or len(df) < 65:
        return {"benchmark": "^TWII", "regime": "UNKNOWN", "ret20": 0.0, "ret60": 0.0, "price_date": ""}
    close = df["Close"]
    p = float(close.iloc[-1])
    ma20 = float(close.tail(20).mean())
    ma60 = float(close.tail(60).mean())
    ret20 = p / float(close.iloc[-21]) - 1.0 if len(close) >= 21 else 0.0
    ret60 = p / float(close.iloc[-61]) - 1.0 if len(close) >= 61 else ret20
    if p >= ma20 and ma20 >= ma60:
        regime = "BULL"
    elif p < ma20 and p >= ma60:
        regime = "NEUTRAL"
    else:
        regime = "BEAR"
    return {
        "benchmark": "^TWII",
        "regime": regime,
        "ret20": ret20,
        "ret60": ret60,
        "price": p,
        "ma20": ma20,
        "ma60": ma60,
        "price_date": str(df.index[-1].date()),
    }


def _pre_score(df: pd.DataFrame, market_ret20: float) -> float:
    if df.empty or len(df) < 130:
        return -999.0
    close = df["Close"]
    vol = df["Volume"]
    p = float(close.iloc[-1])
    r20 = p / float(close.iloc[-21]) - 1.0 if len(close) >= 21 else 0.0
    r60 = p / float(close.iloc[-61]) - 1.0 if len(close) >= 61 else r20
    r120 = p / float(close.iloc[-121]) - 1.0 if len(close) >= 121 else r60
    ma20 = float(close.tail(20).mean())
    ma60 = float(close.tail(60).mean())
    vr = float(vol.tail(5).mean()) / max(1.0, float(vol.tail(20).mean()))
    rs = r20 - market_ret20
    trend = (1.0 if p > ma20 else 0.0) + (1.0 if ma20 > ma60 else 0.0)
    return 100.0 * (0.28 * np.tanh(rs * 5) + 0.24 * np.tanh(r60 * 2.5) + 0.18 * np.tanh(r120 * 1.5) + 0.12 * np.tanh((vr - 1) * 1.4) + 0.09 * trend)



def _prefilter_snapshot(df: pd.DataFrame, market_ret20: float, market_ret60: float) -> dict:
    """Cheap recent-price snapshot used for sector leadership.

    This deliberately uses only the same ~140 trading days already loaded by
    the liquidity prefilter, so adding sector breadth does not add network calls.
    """
    if df.empty or len(df) < 65:
        return {}
    close = pd.to_numeric(df["Close"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce")
    p = float(close.iloc[-1])
    r20 = p / float(close.iloc[-21]) - 1.0 if len(close) >= 21 and float(close.iloc[-21]) > 0 else 0.0
    r60 = p / float(close.iloc[-61]) - 1.0 if len(close) >= 61 and float(close.iloc[-61]) > 0 else r20
    ma20 = float(close.tail(20).mean())
    ma60 = float(close.tail(60).mean())
    vol5 = float(volume.tail(5).mean())
    vol20 = float(volume.tail(20).mean())
    return {
        "ret20": r20,
        "ret60": r60,
        "excess20": r20 - float(market_ret20 or 0.0),
        "excess60": r60 - float(market_ret60 or 0.0),
        "above20": 1.0 if p >= ma20 else 0.0,
        "above60": 1.0 if p >= ma60 else 0.0,
        "volume_active": 1.0 if vol5 >= vol20 * 1.05 else 0.0,
        "vol_ratio": vol5 / max(1.0, vol20),
    }


def _build_sector_leadership(liquid: list[dict]) -> dict[str, dict]:
    """Build a dynamic sector-strength map from the current market itself.

    No sector is hard-coded as the 'main theme'. A sector earns a high score only
    when its median stock is outperforming the index *and* breadth confirms that
    the move is not carried by one or two names.
    """
    grouped: dict[str, list[dict]] = {}
    for row in liquid:
        industry = str(row.get("industry") or "未分類")
        feat = row.get("prefilter") or {}
        if feat:
            grouped.setdefault(industry, []).append(feat)
    out: dict[str, dict] = {}
    for industry, rows in grouped.items():
        n = len(rows)
        if n < 3:
            out[industry] = {"score": 50.0, "count": n, "status": "樣本少"}
            continue
        ex20 = float(np.median([float(r.get("excess20") or 0.0) for r in rows]))
        ex60 = float(np.median([float(r.get("excess60") or 0.0) for r in rows]))
        breadth20 = float(np.mean([float(r.get("above20") or 0.0) for r in rows]))
        breadth60 = float(np.mean([float(r.get("above60") or 0.0) for r in rows]))
        active = float(np.mean([float(r.get("volume_active") or 0.0) for r in rows]))
        # 50 = market-like. RS/persistence matter most; breadth and participation
        # prevent one hot stock from making an entire industry look like a leader.
        score = 50.0
        score += 18.0 * np.tanh(ex20 * 6.0)
        score += 14.0 * np.tanh(ex60 * 3.5)
        score += 10.0 * (breadth20 - 0.5) * 2.0
        score += 6.0 * (breadth60 - 0.5) * 2.0
        score += 4.0 * (active - 0.5) * 2.0
        score = float(np.clip(score, 15.0, 90.0))
        status = "主流" if score >= 68 else "偏強" if score >= 58 else "整理" if score >= 43 else "偏弱"
        out[industry] = {
            "score": round(score, 1),
            "count": n,
            "status": status,
            "excess20_pct": round(ex20 * 100.0, 2),
            "excess60_pct": round(ex60 * 100.0, 2),
            "breadth20_pct": round(breadth20 * 100.0, 1),
            "breadth60_pct": round(breadth60 * 100.0, 1),
            "active_volume_pct": round(active * 100.0, 1),
        }
    return out


def _build_theme_leadership(liquid: list[dict]) -> dict[str, dict]:
    """Dynamic supply-chain breadth using stable theme buckets.

    This complements official-industry breadth. A theme needs at least three
    liquid names; otherwise it is neutral, preventing one famous stock from
    manufacturing a narrative score.
    """
    staged = []
    for row in liquid:
        theme = str(row.get("theme") or "未分類")
        staged.append({"industry": theme, "prefilter": row.get("prefilter") or {}})
    return _build_sector_leadership(staged)


def _sector_score(stock: dict) -> float:
    try:
        return float((stock.get("sector_strength") or {}).get("score") or 50.0)
    except Exception:
        return 50.0


def _theme_score(stock: dict) -> float:
    try:
        return float((stock.get("theme_strength") or {}).get("score") or 50.0)
    except Exception:
        return 50.0


def _mainline_score(stock: dict) -> float:
    """Blend official-industry breadth and supply-chain breadth.

    Official sectors are more robust and receive slightly more weight; the
    supply-chain layer catches cross-industry themes such as semiconductor fab
    build-out without turning the model into a narrative picker.
    """
    return float(np.clip(0.56 * _sector_score(stock) + 0.44 * _theme_score(stock), 15.0, 90.0))

def _finite_list(values) -> list[float]:
    out = []
    for v in values or []:
        try:
            f = float(v)
            if np.isfinite(f):
                out.append(f)
        except Exception:
            continue
    return out


def _fundamental_score(research: dict, industry: str = "", horizon: str = "mid") -> tuple[float | None, dict]:
    """Robust company-evidence score.

    V13.4 intentionally separates *growth*, *quality* and *valuation* instead
    of letting one explosive revenue month dominate.  Financial companies are
    treated separately because their reported monthly revenue/gross margin are
    not comparable with manufacturers.
    """
    if not research:
        return None, {}
    rev = research.get("monthly_revenue") or {}
    fin = research.get("financials") or {}
    val = research.get("valuation") or {}
    detail: dict = {}
    is_financial = "金融" in str(industry)

    # ---------- revenue growth: persistence > single-month spike ----------
    revenue_score = None
    latest_yoy = avg3 = median6 = prior3 = None
    yoy6: list[float] = []
    latest_mom = None
    rows = rev.get("rows") or []
    if rev.get("available") and rows and not is_financial:
        recent = rows[-6:]
        yoy6 = _finite_list([r.get("yoy_pct") for r in recent])
        latest = rows[-1]
        try:
            latest_yoy = float(latest.get("yoy_pct")) if latest.get("yoy_pct") is not None else None
        except Exception:
            latest_yoy = None
        try:
            latest_mom = float(latest.get("mom_pct")) if latest.get("mom_pct") is not None else None
        except Exception:
            latest_mom = None
        avg3_vals = _finite_list([r.get("yoy_pct") for r in rows[-3:]])
        prior3_vals = _finite_list([r.get("yoy_pct") for r in rows[-6:-3]])
        avg3 = float(np.mean(avg3_vals)) if avg3_vals else None
        prior3 = float(np.mean(prior3_vals)) if prior3_vals else None
        median6 = float(np.median(yoy6)) if yoy6 else None
        positive_ratio = float(np.mean(np.array(yoy6) > 0)) if yoy6 else None

        parts = []
        if latest_yoy is not None:
            parts.append((float(np.clip(50 + latest_yoy * 0.65, 0, 100)), 0.22))
        if avg3 is not None:
            parts.append((float(np.clip(50 + avg3 * 0.75, 0, 100)), 0.28))
        if median6 is not None:
            parts.append((float(np.clip(50 + median6 * 0.65, 0, 100)), 0.24))
        if positive_ratio is not None:
            parts.append((float(np.clip(25 + positive_ratio * 75, 0, 100)), 0.16))
        if avg3 is not None and prior3 is not None:
            parts.append((float(np.clip(50 + (avg3 - prior3) * 0.8, 15, 90)), 0.10))
        if parts:
            w = sum(x[1] for x in parts)
            revenue_score = sum(x[0] * x[1] for x in parts) / w

            # One-off spike penalty. A sudden +100% month following a weak base
            # is interesting, but not equivalent to six months of sustained growth.
            spike_penalty = 0.0
            if latest_yoy is not None and median6 is not None and latest_yoy - median6 > 45:
                spike_penalty += min(10.0, (latest_yoy - median6 - 45) * 0.12)
            if latest_mom is not None and abs(latest_mom) > 55:
                spike_penalty += min(5.0, (abs(latest_mom) - 55) * 0.08)
            revenue_score = float(np.clip(revenue_score - spike_penalty, 0, 100))
            detail.update({
                "revenue_score": round(revenue_score, 1),
                "revenue_latest_yoy": round(latest_yoy, 2) if latest_yoy is not None else None,
                "revenue_avg3_yoy": round(avg3, 2) if avg3 is not None else None,
                "revenue_median6_yoy": round(median6, 2) if median6 is not None else None,
                "revenue_positive_ratio6": round(positive_ratio, 2) if positive_ratio is not None else None,
                "revenue_spike_penalty": round(spike_penalty, 1),
            })

        # Official OpenAPI fallback can contain only the latest month. It is useful
        # evidence, but deliberately receives lower confidence than 3-6 month history.
        if revenue_score is None and latest_yoy is not None:
            revenue_score = float(np.clip(50.0 + latest_yoy * 0.55, 20.0, 85.0))
            if latest_mom is not None:
                revenue_score = 0.80 * revenue_score + 0.20 * float(np.clip(50.0 + latest_mom * 0.7, 20.0, 80.0))
            detail.update({
                "revenue_score": round(revenue_score, 1),
                "revenue_latest_yoy": round(latest_yoy, 2),
                "revenue_history_limited": True,
            })

    # ---------- earnings / margin quality ----------
    earnings_score = margin_score = quality_score = None
    eps: list[float] = []
    gm: list[float] = []
    if fin.get("available"):
        qs = fin.get("quarters") or []
        eps = _finite_list([q.get("eps") for q in qs])
        gm = _finite_list([q.get("gross_margin_pct") for q in qs])
        if eps:
            pos_ratio = float(np.mean(np.array(eps) > 0))
            eps4 = float(np.sum(eps))
            earnings_score = 25.0 + pos_ratio * 45.0
            earnings_score += 10.0 if eps[-1] > 0 else -15.0
            if len(eps) >= 2:
                earnings_score += 8.0 if eps[-1] > eps[-2] else -4.0
            if len(eps) >= 4:
                recent2 = float(np.mean(eps[-2:]))
                prior2 = float(np.mean(eps[-4:-2]))
                scale = max(0.5, abs(prior2))
                earnings_score += float(np.clip((recent2 - prior2) / scale * 10.0, -12.0, 12.0))
            earnings_score += 5.0 if eps4 > 0 else -15.0
            earnings_score = float(np.clip(earnings_score, 0, 100))
            detail.update({
                "earnings_score": round(earnings_score, 1),
                "eps_positive_ratio4": round(pos_ratio, 2),
                "eps_4q_sum": round(eps4, 2),
            })
        if len(gm) >= 2 and not is_financial:
            gm_arr = np.array(gm, dtype=float)
            margin_score = 55.0 + float(np.clip((gm[-1] - gm[0]) * 2.5, -20, 20))
            margin_score -= min(15.0, float(np.std(gm_arr)) * 1.8)
            if gm[-1] > 0:
                margin_score += 8.0
            else:
                margin_score -= 25.0
            margin_score = float(np.clip(margin_score, 0, 100))
            detail["margin_score"] = round(margin_score, 1)

    # Latest official cumulative EPS fallback (not a 4-quarter sum). Keep the
    # score conservative because historical earnings persistence is unavailable.
    if earnings_score is None and fin.get("available") and fin.get("latest_ytd_eps") is not None:
        latest_ytd_eps = finite_scalar(fin.get("latest_ytd_eps"), 0.0)
        earnings_score = float(np.clip(52.0 + np.tanh(latest_ytd_eps / 5.0) * 22.0, 25.0, 74.0))
        detail.update({
            "earnings_score": round(earnings_score, 1),
            "latest_ytd_eps": round(latest_ytd_eps, 2),
            "financial_history_limited": True,
        })
    if margin_score is None and fin.get("available") and fin.get("gross_margin_latest_pct") is not None and not is_financial:
        gm_latest = finite_scalar(fin.get("gross_margin_latest_pct"), 0.0)
        margin_score = float(np.clip(48.0 + gm_latest * 0.8, 20.0, 75.0))
        detail["margin_score"] = round(margin_score, 1)

    quality_parts = []
    if earnings_score is not None:
        quality_parts.append((earnings_score, 0.68 if not is_financial else 1.0))
    if margin_score is not None:
        quality_parts.append((margin_score, 0.32))
    if quality_parts:
        qw = sum(w for _, w in quality_parts)
        quality_score = sum(s * w for s, w in quality_parts) / qw
        detail["quality_score"] = round(float(quality_score), 1)

    # ---------- valuation sanity, growth-adjusted ----------
    valuation_score = None
    pe = (val or {}).get("pe")
    try:
        pe = float(pe) if pe is not None else None
    except Exception:
        pe = None
    if pe is not None and np.isfinite(pe):
        if pe <= 0:
            valuation_score = 18.0
        elif is_financial:
            valuation_score = float(np.clip(88.0 - abs(pe - 14.0) * 2.0, 35.0, 90.0))
        else:
            growth_ref = max(0.0, avg3 or median6 or 0.0)
            fair_band = 24.0 + min(46.0, growth_ref * 0.75)
            if pe <= fair_band:
                valuation_score = float(np.clip(72.0 + (fair_band - pe) * 0.25, 55.0, 88.0))
            else:
                valuation_score = float(np.clip(72.0 - (pe - fair_band) * 0.65, 25.0, 72.0))
        detail["valuation_score"] = round(valuation_score, 1)

    # ---------- explicit red flags: evidence can veto a momentum mirage ----------
    red_flag_penalty = 0.0
    if eps and float(np.sum(eps)) <= 0:
        red_flag_penalty += 16.0
    if gm and not is_financial and gm[-1] < 0:
        red_flag_penalty += 18.0
    if latest_yoy is not None and avg3 is not None and latest_yoy <= -35 and avg3 <= -20:
        red_flag_penalty += 16.0
    if yoy6 and float(np.mean(np.array(yoy6) > 0)) <= 0.25 and float(np.median(yoy6)) < -15:
        red_flag_penalty += 8.0
    red_flag_penalty = float(min(35.0, red_flag_penalty))
    detail["fundamental_red_flag_penalty"] = round(red_flag_penalty, 1)
    detail["financial_sector_method"] = bool(is_financial)

    # Financials: monthly revenue is accounting-noisy, so EPS + valuation carry
    # the company score. Other sectors blend persistent growth and quality.
    components = []
    if is_financial:
        if quality_score is not None:
            components.append((quality_score, 0.72 if horizon == "long" else 0.65))
        if valuation_score is not None:
            components.append((valuation_score, 0.28 if horizon == "long" else 0.35))
    else:
        h_weights = {
            "short": {"growth": 0.50, "quality": 0.30, "valuation": 0.20},
            "mid": {"growth": 0.47, "quality": 0.38, "valuation": 0.15},
            "long": {"growth": 0.30, "quality": 0.50, "valuation": 0.20},
        }.get(horizon, {"growth": 0.47, "quality": 0.38, "valuation": 0.15})
        if revenue_score is not None:
            components.append((revenue_score, h_weights["growth"]))
        if quality_score is not None:
            components.append((quality_score, h_weights["quality"]))
        if valuation_score is not None:
            components.append((valuation_score, h_weights["valuation"]))

    if not components:
        return None, detail
    wsum = sum(w for _, w in components)
    score = sum(s * w for s, w in components) / wsum
    # Keep red flags visible in both the company score and the final rank.
    score -= red_flag_penalty * (0.45 if horizon == "short" else 0.70 if horizon == "mid" else 0.85)
    detail["fundamental_score_raw"] = round(float(np.clip(score, 0, 100)), 1)
    return round(float(np.clip(score, 0, 100)), 1), detail


def _liquidity_score(avg_turnover: float) -> float:
    """0-100 execution-quality proxy from 20-day average turnover.

    This is deliberately small in the final model: it prevents very thin names
    from tying with equally strong liquid names, but does not turn the strategy
    into a large-cap index tracker.
    """
    try:
        t = max(1.0, float(avg_turnover))
    except Exception:
        return 0.0
    # 10m ~= 0, 100m ~= 33, 1b ~= 67, 10b+ ~= 100
    return float(np.clip((math.log10(t) - 7.0) / 3.0 * 100.0, 0.0, 100.0))


def _flow_score(research: dict, avg_volume_shares: float) -> tuple[float | None, dict]:
    if not research:
        return None, {}
    inst = research.get("institutional_flow") or {}
    if not inst.get("available"):
        return None, {}
    net_lots = float(inst.get("total_net_lots") or 0.0)
    monthly_lots = max(1.0, avg_volume_shares / 1000.0 * max(1, int(inst.get("sessions") or 20)))
    ratio = net_lots / monthly_lots
    score = float(np.clip(50.0 + ratio * 180.0, 5.0, 95.0))
    detail = {"institutional_net_to_volume": round(ratio, 4)}

    branch = research.get("main_force_proxy") or {}
    if branch.get("available"):
        proxy = float(branch.get("proxy_net_lots") or 0.0)
        branch_ratio = proxy / max(1.0, avg_volume_shares / 1000.0 * max(1, int(branch.get("sessions") or 10)))
        branch_score = float(np.clip(50.0 + branch_ratio * 120.0, 5.0, 95.0))
        score = 0.72 * score + 0.28 * branch_score
        detail["branch_proxy_net_to_volume"] = round(branch_ratio, 4)
    return round(float(np.clip(score, 0, 100)), 1), detail


def _forecast_score(forecast: dict) -> float | None:
    if not forecast or not forecast.get("estimate_available"):
        return None
    # V13.3 return engine already combines analog median, downside, sample
    # de-overlap and a smoothed positive-rate into one empirical quality score.
    # Use that directly so the ranking does not double-count the same outcome.
    quality = forecast.get("empirical_quality_score")
    if quality is not None:
        try:
            return round(float(np.clip(float(quality), 0, 100)), 1)
        except Exception:
            pass
    s = forecast.get("strategy") or {}
    median = float(s.get("median") or 0.0)
    p10 = float(s.get("p10") or 0.0)
    reward = 50.0 + np.tanh(median * 8.0) * 30.0
    penalty = max(0.0, -p10 - 0.05) * 80.0
    return round(float(np.clip(reward - penalty, 0, 100)), 1)


def _combined_ranking_score(stock: dict, horizon: str, model_family: str) -> None:
    block = stock["horizons"][horizon]
    forecast = block.get("forecast") or {}
    technical = float(forecast.get("technical_factor_score", forecast.get("composite_factor_score", 0.0)) or 0.0)
    empirical = _forecast_score(forecast)
    fundamental, f_detail = _fundamental_score(
        stock.get("research") or {},
        industry=str(stock.get("industry") or ""),
        horizon=horizon,
    )
    flow, flow_detail = _flow_score(stock.get("research") or {}, float(stock.get("avg_volume_20d") or 0.0))
    liquidity = _liquidity_score(float(stock.get("avg_turnover_20d") or 0.0))

    sector = _sector_score(stock)
    theme = _theme_score(stock)
    mainline = _mainline_score(stock)
    weights = {
        # V16 uses an objective industry-leadership layer. It is deliberately
        # capped below the stock-specific evidence, so the model can follow a
        # genuine market mainline without blindly buying every stock in a hot theme.
        "short": {"technical": 0.36, "empirical": 0.17, "fundamental": 0.07, "flow": 0.18, "liquidity": 0.07, "mainline": 0.15},
        "mid": {"technical": 0.27, "empirical": 0.18, "fundamental": 0.27, "flow": 0.10, "liquidity": 0.04, "mainline": 0.14},
        "long": {"technical": 0.20, "empirical": 0.18, "fundamental": 0.43, "flow": 0.05, "liquidity": 0.03, "mainline": 0.11},
    }[horizon]

    allowed = {"technical", "empirical"}
    if model_family in {"business_confirmed", "full"}:
        allowed.add("fundamental")
    if model_family in {"flow_confirmed", "full"}:
        allowed.add("flow")

    allowed.add("liquidity")
    allowed.add("mainline")
    values = {
        "technical": technical,
        "empirical": empirical,
        "fundamental": fundamental,
        "flow": flow,
        "liquidity": liquidity,
        "mainline": mainline,
    }
    present = [(k, values[k], weights[k]) for k in allowed if values[k] is not None]
    if not present:
        rank = technical
    else:
        wsum = sum(w for _, _, w in present)
        rank = sum(float(v) * w for _, v, w in present) / wsum

    # Mainline resonance is earned only when sector breadth and company/stock
    # evidence agree. This avoids the narrative trap of assuming 'hot industry'
    # automatically means every constituent is a good investment.
    resonance_bonus = 0.0
    if mainline >= 68 and technical >= 62:
        if horizon == "short" and (flow is None or flow >= 52):
            resonance_bonus = min(3.0, (mainline - 68) * 0.08 + (technical - 62) * 0.04)
        elif horizon == "mid" and fundamental is not None and fundamental >= 55:
            resonance_bonus = min(3.0, (mainline - 68) * 0.07 + (fundamental - 55) * 0.04)
        elif horizon == "long" and fundamental is not None and fundamental >= 62:
            resonance_bonus = min(2.5, (mainline - 68) * 0.05 + (fundamental - 62) * 0.04)
    if mainline < 35:
        resonance_bonus -= {"short": 1.5, "mid": 2.0, "long": 1.0}[horizon]
    rank += resonance_bonus

    # Entry timing is not a hard gate, but an already-extended stock should not
    # receive the same recommendation rank as an equally strong stock near a
    # reasonable entry.  V13.3.1 computed an entry score but did not feed it
    # into ranking; V13.3.2 uses only a modest penalty so rank-first remains.
    plan = block.get("plan") or {}
    entry_score = plan.get("entry_score")
    entry_penalty = 0.0
    if entry_score is not None:
        try:
            es = float(entry_score)
            if es < 70.0:
                scale = {"short": 0.40, "mid": 0.22, "long": 0.10}[horizon]
                entry_penalty += min({"short": 8.0, "mid": 4.0, "long": 2.0}[horizon], (70.0 - es) * scale)
        except Exception:
            pass
    if block.get("entry_state") == "DO_NOT_CHASE":
        entry_penalty += {"short": 10.0, "mid": 5.0, "long": 2.0}[horizon]
    rank -= entry_penalty

    # Evidence quality: missing data is not treated as a zero, but a full-model
    # recommendation with no company evidence should not be equally confident
    # as a similarly ranked stock whose company data confirms the thesis.
    evidence_penalty = 0.0
    if model_family in {"business_confirmed", "full"} and fundamental is None:
        evidence_penalty += {"short": 1.5, "mid": 4.0, "long": 7.0}[horizon]
    if empirical is None:
        evidence_penalty += {"short": 1.5, "mid": 3.0, "long": 5.0}[horizon]
    if model_family in {"flow_confirmed", "full"} and flow is None:
        evidence_penalty += {"short": 2.0, "mid": 1.5, "long": 0.5}[horizon]

    # When the data positively identifies collapsing operations / persistent
    # losses, apply a second, horizon-aware guardrail. This is intentionally
    # mild for short trades and strong for long recommendations.
    red_flag = float(f_detail.get("fundamental_red_flag_penalty") or 0.0)
    red_guard = red_flag * {"short": 0.08, "mid": 0.28, "long": 0.45}[horizon]
    rank -= evidence_penalty + red_guard

    # Gentle market-regime adaptation is performed in run_scan by adding a small modifier.
    block["ranking_score"] = round(float(np.clip(rank, 0, 100)), 1)
    block["score_components"] = {
        "entry_penalty": round(entry_penalty, 1),
        "evidence_penalty": round(evidence_penalty, 1),
        "red_flag_guard": round(red_guard, 1),
        "technical": round(technical, 1),
        "empirical": empirical,
        "fundamental": fundamental,
        "flow": flow,
        "liquidity": round(liquidity, 1),
        "sector": round(sector, 1),
        "theme": round(theme, 1),
        "mainline": round(mainline, 1),
        "mainline_resonance": round(resonance_bonus, 1),
        **f_detail,
        **flow_detail,
    }



def _decorate_role_stage(stock: dict) -> None:
    """Attach the original three-layer portfolio role and a plain-language stage.

    V13.3 keeps the user's original Core / Next Compounder / Tactical pyramid,
    but computes it from current evidence rather than hard-coding company names.
    The role is descriptive and never overrides the horizon ranking.
    """
    horizons = stock.get("horizons") or {}
    short = horizons.get("short") or {}
    mid = horizons.get("mid") or {}
    long = horizons.get("long") or {}
    short_score = float(short.get("ranking_score") or 0.0)
    mid_score = float(mid.get("ranking_score") or 0.0)
    long_score = float(long.get("ranking_score") or 0.0)
    fscore = (mid.get("score_components") or {}).get("fundamental")
    fscore = float(fscore) if fscore is not None else 50.0
    flow = (short.get("score_components") or {}).get("flow")
    flow = float(flow) if flow is not None else 50.0

    forecast = short.get("forecast") or {}
    feat = forecast.get("features") or {}
    vol_ratio = float(feat.get("vol_ratio") or 1.0)
    alpha = float(forecast.get("alpha_mean") or 0.0)
    plan = short.get("plan") or {}
    price = float(stock.get("price") or 0.0)
    ma5 = float(plan.get("ma5") or price or 1.0)
    ma20 = float(plan.get("ma20") or price or 1.0)
    ma60 = float(plan.get("ma60") or ma20)
    ma120 = float(plan.get("ma120") or ma60)
    gap20 = price / ma20 - 1.0 if ma20 > 0 else 0.0

    research = stock.get("research") or {}
    rev = research.get("monthly_revenue") or {}
    rev_rows = rev.get("rows") or []
    latest_yoy = None
    latest_mom = None
    if rev_rows:
        latest_yoy = rev_rows[-1].get("yoy_pct")
        latest_mom = rev_rows[-1].get("mom_pct")
    avg3_yoy = rev.get("avg_yoy_3m_pct")

    fin = research.get("financials") or {}
    qs = fin.get("quarters") or []
    eps = [float(q.get("eps")) for q in qs if q.get("eps") is not None]
    gm = [float(q.get("gross_margin_pct")) for q in qs if q.get("gross_margin_pct") is not None]
    eps_accel = len(eps) >= 2 and eps[-1] > max(0.0, eps[-2] * 1.08)
    margin_stable = len(gm) >= 2 and gm[-1] >= gm[-2] - 1.0
    revenue_turn = bool(latest_yoy is not None and float(latest_yoy) > 5.0)
    revenue_accel = bool(
        latest_yoy is not None and avg3_yoy is not None
        and float(latest_yoy) >= max(8.0, float(avg3_yoy))
    )
    fundamental_turn = bool(revenue_turn and eps_accel)

    # Stage follows the original S1-S5 idea but is shown in plain language.
    if gap20 >= 0.20:
        stage_code, stage_label = 5, "過熱"
    elif gap20 >= 0.12:
        stage_code, stage_label = 4, "偏離過大"
    elif price > ma5 > ma20 > ma60 > ma120:
        stage_code, stage_label = 3, "趨勢成形"
    elif stock.get("setup") == "BREAKOUT" and vol_ratio >= 1.2 and alpha > 0:
        stage_code, stage_label = 2, "突破"
    elif fundamental_turn:
        stage_code, stage_label = 1, "基本面轉折"
    else:
        stage_code, stage_label = 0, "整理"

    # Original pyramid scoring, made data-driven:
    # Core: quality/leadership + RS + stability + above MA20.
    # Growth: fundamental turn/acceleration + RS + breakout/volume.
    # Tactical: RS + breakout + volume + above MA5.
    atr_pct = float(feat.get("atr_pct") or 0.03)
    stability = float(np.clip(100.0 - max(0.0, atr_pct - 0.012) * 1500.0, 20.0, 95.0))
    leadership_bonus = 25.0 if long_score >= 78 and fscore >= 65 else 15.0 if long_score >= 70 else 0.0
    defensive_bonus = 20.0 if stability >= 78 and fscore >= 60 and margin_stable else 0.0
    quality_bonus = max(leadership_bonus, defensive_bonus)
    core_rs_bonus = float(np.clip(alpha * 200.0, -10.0, 20.0))
    stability_bonus = float(np.clip((stability - 50.0) / 45.0 * 20.0, 0.0, 20.0))
    above20_bonus = 15.0 if price >= ma20 else 0.0
    core = float(np.clip(50.0 + quality_bonus + core_rs_bonus + stability_bonus + above20_bonus, 0.0, 98.0))

    growth_rs_bonus = float(np.clip(alpha * 150.0, -10.0, 15.0))
    growth = 50.0
    growth += 25.0 if fundamental_turn else 0.0
    growth += 20.0 if revenue_accel else 0.0
    growth += 20.0 if eps_accel else 0.0
    growth += growth_rs_bonus
    growth += 10.0 if stock.get("setup") == "BREAKOUT" else 0.0
    growth += 10.0 if vol_ratio > 1.2 else 0.0
    # If company evidence is unavailable, anchor growth to the mid-horizon model
    # rather than pretending the missing fundamental bonuses are zero evidence.
    if not (rev.get("available") or fin.get("available")):
        growth = 0.70 * mid_score + 0.30 * (50.0 + growth_rs_bonus)
    growth = float(np.clip(growth, 0.0, 98.0))

    tactical_rs_bonus = float(np.clip(alpha * 200.0, -10.0, 30.0))
    tactical = 50.0 + tactical_rs_bonus
    tactical += 25.0 if stock.get("setup") == "BREAKOUT" else 18.0 if stock.get("setup") in {"WASHOUT", "RECLAIM"} else 0.0
    tactical += 20.0 if vol_ratio > 1.3 else 0.0
    tactical += 15.0 if price >= ma5 else 0.0
    # Keep the role connected to the actual short-horizon ranking.
    tactical = 0.75 * float(np.clip(tactical, 0.0, 98.0)) + 0.25 * short_score
    tactical = float(np.clip(tactical, 0.0, 98.0))

    role_scores = {"core": core, "growth": growth, "tactical": tactical}
    role = max(role_scores, key=role_scores.get)
    role_label = {"core": "核心型", "growth": "成長型", "tactical": "短打型"}[role]

    stock["stage"] = {"code": stage_code, "label": stage_label}
    stock["role"] = {
        "code": role, "label": role_label,
        "scores": {k: round(v, 1) for k, v in role_scores.items()},
    }


def _position_guidance(regime: str, horizon: str) -> dict:
    """Regime-aware exposure hint; it scales conviction, not forecast returns."""
    table = {
        "BULL": {"short": 1.00, "mid": 0.95, "long": 0.95},
        "NEUTRAL": {"short": 0.70, "mid": 0.80, "long": 0.85},
        "BEAR": {"short": 0.40, "mid": 0.55, "long": 0.70},
        "UNKNOWN": {"short": 0.60, "mid": 0.70, "long": 0.75},
    }
    ratio = float(table.get(str(regime), table["UNKNOWN"]).get(horizon, 0.70))
    if ratio >= 0.90:
        label = "一般部位"
    elif ratio >= 0.70:
        label = "稍微保守"
    elif ratio >= 0.50:
        label = "縮小部位"
    else:
        label = "小部位"
    return {"ratio": ratio, "label": label}

def _research_availability(research: dict | None) -> dict:
    r = research or {}
    return {
        "revenue": bool((r.get("monthly_revenue") or {}).get("available")),
        "financials": bool((r.get("financials") or {}).get("available")),
        "valuation": bool((r.get("valuation") or {}).get("available")),
        "institutional_flow": bool((r.get("institutional_flow") or {}).get("available")),
        "main_force_proxy": bool((r.get("main_force_proxy") or {}).get("available")),
    }


def _has_useful_research(research: dict | None) -> bool:
    a = _research_availability(research)
    return any(a.values())


def _enrich_tickers(
    stocks: list[dict], data_dir: Path, tickers: list[str], progress=None,
    progress_value: float = 0.88, progress_end: float | None = None,
):
    pool = [t for t in dict.fromkeys(tickers) if t]
    if not pool:
        return
    by_ticker = {s["ticker"]: s for s in stocks}
    end_value = float(progress_end if progress_end is not None else min(0.95, progress_value + 0.04))
    if progress:
        progress(f"公司資料 0/{len(pool)}", progress_value)

    def one(ticker):
        client = ResearchDataClient(data_dir / "research_cache.sqlite")
        return ticker, client.stock_research(ticker, include_branch=False)

    workers = min(6, max(1, len(pool)))
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(one, ticker) for ticker in pool]
        for fut in as_completed(futures):
            completed += 1
            try:
                ticker, research = fut.result()
                if ticker in by_ticker:
                    by_ticker[ticker]["research"] = research
            except Exception:
                pass
            if progress:
                v = progress_value + (end_value - progress_value) * completed / max(1, len(pool))
                progress(f"公司資料 {completed}/{len(pool)}", v)


def _enrich_research(stocks: list[dict], data_dir: Path, settings: RunSettings, progress=None):
    if not stocks:
        return
    tickers = []
    for h in ["short", "mid", "long"]:
        ranked = sorted(
            stocks,
            key=lambda x: x.get("horizons", {}).get(h, {}).get("forecast", {}).get("composite_factor_score", 0),
            reverse=True,
        )[: max(5, int(settings.research_pool_per_horizon))]
        tickers.extend([x["ticker"] for x in ranked])

    # V13.4 quality probe: pure technical pre-ranking can miss a liquid, steady
    # leader whose recent momentum is less explosive than small hot stocks. Add
    # a small high-turnover probe without expanding research calls to the full
    # 1,000-stock candidate set. This is a discovery aid, not a ranking bonus.
    liquid_probe = sorted(
        stocks,
        key=lambda x: float(x.get("avg_turnover_20d") or 0.0),
        reverse=True,
    )[:8]
    tickers.extend([x["ticker"] for x in liquid_probe])
    _enrich_tickers(stocks, data_dir, tickers, progress=progress, progress_value=0.84, progress_end=0.90)

def run_scan(data_dir: Path, settings: RunSettings, progress=None) -> dict:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if progress:
        progress("載入上市櫃股票", 0.05)
    universe = fetch_twse_universe()
    if universe is None or len(universe) < MIN_PRODUCTION_UNIVERSE:
        raise RuntimeError(f"股票母體僅取得 {0 if universe is None else len(universe)} 檔；為避免以不完整市場資料覆蓋舊快照，本次更新已中止")
    tickers = universe["ticker"].astype(str).tolist()

    store = DailyPriceStore(data_dir / "daily_prices.sqlite")
    if progress:
        progress("準備行情快取", 0.14)

    def price_progress(done, total, message):
        if not progress:
            return
        frac = float(done) / max(1.0, float(total))
        progress(str(message), 0.16 + 0.26 * min(1.0, max(0.0, frac)))

    dl = store.batch_fetch_and_update(
        tickers + ["^TWII"], period=settings.history_period, progress=price_progress
    )
    if progress:
        mode_text = "沿用多年快取" if dl.get("mode") == "incremental" else "歷史資料更新完成"
        progress(mode_text + "，核對官方最新交易日", 0.43)
    # Resolve the latest *complete* Taiwan trading session first.  Yahoo often
    # labels the newest Taiwan bar one session late on cloud deployments, so the
    # exchange close is authoritative and Yahoo is only the historical backbone.
    cal = calendar_reference(data_dir, now=_taipei_timestamp())
    calendar_expected_date = str(cal.get("expected_date") or "")
    def official_progress(done, total, message):
        if progress:
            progress(str(message), 0.43 + 0.05 * float(done) / max(1.0, float(total)))

    official_eod = store.overlay_official_eod(target_date=calendar_expected_date, progress=official_progress)
    expected_date = calendar_expected_date
    # Weekday public holidays are not inferable from weekday arithmetic alone.
    # If the date-specific TWSE endpoint responds successfully but explicitly
    # has no data, trust the exchange's latest session date instead of excluding
    # the whole market as stale. A network failure does not trigger this fallback.
    if (
        calendar_expected_date
        and official_eod.get("twse_target_response")
        and not official_eod.get("twse_target_has_data")
        and official_eod.get("twse_date")
        and str(official_eod.get("twse_date")) < calendar_expected_date
    ):
        expected_date = str(official_eod.get("twse_date"))
        official_eod["calendar_expected_date"] = calendar_expected_date
        official_eod["exchange_calendar_adjusted"] = True
    market = _market_context(store)
    market_ret20 = float(market.get("ret20") or 0.0)
    market_ret60 = float(market.get("ret60") or 0.0)
    benchmark_df = store.get_prices("^TWII")

    liquid = []
    valid_count = 0
    stale_count = 0
    if progress:
        progress("篩選成交量足夠的股票", 0.48)
    for _, row in universe.iterrows():
        ticker = str(row["ticker"])
        # Prefilter only needs 120 trading days. Reading all five years for ~2,000
        # stocks here was a major avoidable I/O cost. Full history is loaded only
        # after the candidate pool is formed.
        df = store.get_prices(ticker, limit=140)
        if len(df) < 140:
            continue
        valid_count += 1
        latest_stock_date = str(df.index[-1].date())
        if expected_date and latest_stock_date < expected_date:
            stale_count += 1
            continue
        try:
            p = float(df["Close"].iloc[-1])
            avg_vol = float(df["Volume"].tail(20).mean())
            avg_turnover = float((df["Close"].tail(20) * df["Volume"].tail(20)).mean())
            if p < settings.min_price or avg_turnover < settings.min_avg_turnover:
                continue
            liquid.append(
                {
                    "ticker": ticker,
                    "name": str(row.get("name", ticker)),
                    "industry": str(row.get("industry", "")),
                    "price": p,
                    "price_date": latest_stock_date,
                    "avg_volume_20d": avg_vol,
                    "avg_turnover_20d": avg_turnover,
                    "pre_score": _pre_score(df, market_ret20),
                    "fine_industry": fine_industry(ticker, str(row.get("name", ticker)), str(row.get("industry", ""))),
                    "prefilter": _prefilter_snapshot(df, market_ret20, market_ret60),
                }
            )
        except Exception:
            continue

    if len(liquid) < MIN_PRODUCTION_EVALUATED:
        raise RuntimeError(f"可用且日期一致的股票僅 {len(liquid)} 檔；本次更新已中止並保留上一份成功快照")

    for row in liquid:
        row["theme"] = theme_bucket(row.get("ticker", ""), row.get("name", ""), row.get("industry", ""), row.get("fine_industry", ""))
    sector_map = _build_sector_leadership(liquid)
    theme_map = _build_theme_leadership(liquid)
    for row in liquid:
        ss = sector_map.get(str(row.get("industry") or "未分類"), {"score": 50.0, "status": "整理"})
        ts = theme_map.get(str(row.get("theme") or "未分類"), {"score": 50.0, "status": "整理"})
        row["sector_strength"] = ss
        row["theme_strength"] = {**ts, "name": str(row.get("theme") or "未分類")}
        ml = 0.56 * float(ss.get("score") or 50.0) + 0.44 * float(ts.get("score") or 50.0)
        # Discovery gets only a small mainline nudge; the full mainline factor is
        # applied later in the horizon-specific ranking.
        row["discovery_score"] = float(row.get("pre_score") or 0.0) + (ml - 50.0) * 0.12
    liquid.sort(key=lambda x: x.get("discovery_score", x.get("pre_score", -999)), reverse=True)
    candidates = liquid[: max(50, int(settings.candidate_size))]

    if progress:
        progress(f"歷史比較 0/{len(candidates)}", 0.54)
    evaluated: list[dict] = []
    total_candidates = max(1, len(candidates))
    for idx_c, c in enumerate(candidates, 1):
        df = store.get_prices(c["ticker"])
        if len(df) < 140:
            continue
        if progress and (idx_c == 1 or idx_c % 10 == 0 or idx_c == len(candidates)):
            progress(f"歷史比較 {idx_c}/{len(candidates)}", 0.54 + 0.28 * idx_c / total_candidates)
        horizons = {}
        forecasts = estimate_all_horizons(df, settings, twii_ret_20d=market_ret20, benchmark_df=benchmark_df)
        for h in ["short", "mid", "long"]:
            plan = generate_trade_plan(df, h)
            state = evaluate_entry_state(df, plan)
            est = forecasts[h]
            horizons[h] = {
                "plan": plan,
                "entry_state": state,
                "forecast": est,
                "qualification": {
                    "research_qualified": bool(est.get("composite_factor_score", 0) >= 35),
                    "rank_first": True,
                },
            }
        evaluated.append(
            {
                "ticker": c["ticker"],
                "name": c["name"],
                "industry": c["industry"],
                "fine_industry": c.get("fine_industry") or fine_industry(c["ticker"], c["name"], c["industry"]),
                "theme": c.get("theme") or theme_bucket(c["ticker"], c["name"], c["industry"], c.get("fine_industry", "")),
                "price": c["price"],
                "price_date": c["price_date"],
                "avg_volume_20d": c["avg_volume_20d"],
                "avg_turnover_20d": c["avg_turnover_20d"],
                "pre_score": c.get("pre_score"),
                "discovery_score": c.get("discovery_score"),
                "sector_strength": c.get("sector_strength") or {"score": 50.0, "status": "整理"},
                "theme_strength": c.get("theme_strength") or {"score": 50.0, "status": "整理", "name": c.get("theme") or "未分類"},
                "setup": setup_from_df(df),
                "horizons": horizons,
                "research": {},
                "evidence": {},
            }
        )

    if len(evaluated) < MIN_PRODUCTION_EVALUATED:
        raise RuntimeError(f"完成歷史模型的股票僅 {len(evaluated)} 檔；本次更新已中止並保留上一份成功快照")

    # Always enrich the front of the ranking so the recommendation cards have the requested fields.
    _enrich_research(evaluated, data_dir, settings, progress=progress)

    # Re-rank with chosen evidence family, re-normalizing missing components rather than treating them as zero.
    def apply_scores(stock):
        for h in ["short", "mid", "long"]:
            _combined_ranking_score(stock, h, settings.model_family)
            regime = market.get("regime")
            block = stock["horizons"][h]
            r = float(block.get("ranking_score") or 0.0)
            if regime == "BEAR" and h == "short":
                r -= 4.0
            elif regime == "BULL" and h in {"short", "mid"}:
                r += 2.0
            block["ranking_score"] = round(float(np.clip(r, 0, 100)), 1)
            block["position_guidance"] = _position_guidance(str(regime or "UNKNOWN"), h)

    for stock in evaluated:
        apply_scores(stock)

    # Guarantee that every displayed TOP-5 card has the requested research fields.
    # Because adding fundamentals/flows can reshuffle the ranking, iterate on a
    # top-10 buffer until the front of each horizon is enriched or the ranking
    # stabilizes. This avoids a technically strong but data-empty stock appearing
    # in the final recommendations.
    for pass_idx in range(4):
        front = []
        for h in ["short", "mid", "long"]:
            front.extend([
                s["ticker"] for s in sorted(
                    evaluated,
                    key=lambda x: x["horizons"][h].get("ranking_score", -1),
                    reverse=True,
                )[:10]
            ])
        front = list(dict.fromkeys(front))
        missing = [t for t in front if not _has_useful_research(next((x for x in evaluated if x["ticker"] == t), {}).get("research"))]
        if not missing:
            break
        start_v = 0.90 + pass_idx * 0.012
        _enrich_tickers(evaluated, data_dir, missing, progress=progress, progress_value=start_v, progress_end=min(0.955, start_v + 0.012))
        enriched_set = set(missing)
        for stock in evaluated:
            if stock["ticker"] in enriched_set:
                apply_scores(stock)

    # Broker-branch data is expensive (Sponsor, one date per request).  Only fetch
    # it for stocks that are already in the displayed top-5 union, and only when
    # the operator explicitly enables it.
    if os.getenv("ENABLE_BRANCH_FLOW", "0") == "1":
        top_tickers = []
        for h in ["short", "mid", "long"]:
            top_tickers.extend([s["ticker"] for s in sorted(evaluated, key=lambda x: x["horizons"][h].get("ranking_score", -1), reverse=True)[:5]])
        top_tickers = list(dict.fromkeys(top_tickers))
        by_ticker = {s["ticker"]: s for s in evaluated}
        if progress:
            progress(f"補充券商分點（{len(top_tickers)} 檔）", 0.96)
        def branch_one(ticker):
            client = ResearchDataClient(data_dir / "research_cache.sqlite")
            return ticker, client.branch_main_force_proxy(ticker.split(".")[0], max_sessions=20)
        with ThreadPoolExecutor(max_workers=min(4, max(1, len(top_tickers)))) as ex:
            futs = [ex.submit(branch_one, t) for t in top_tickers]
            for fut in as_completed(futs):
                try:
                    ticker, branch = fut.result()
                    if ticker in by_ticker:
                        by_ticker[ticker].setdefault("research", {})["main_force_proxy"] = branch
                        apply_scores(by_ticker[ticker])
                except Exception:
                    continue

    def refresh_metadata(stock):
        _decorate_role_stage(stock)
        stock["evidence"] = _research_availability(stock.get("research") or {})

    for stock in evaluated:
        refresh_metadata(stock)

    # Crucial production pass: the joint cross-horizon allocator can promote a
    # stock that was outside the raw top-10 research pool. Ensure the actual 15
    # displayed names have company evidence, then re-score/reallocate until stable.
    for pass_idx in range(5):
        temp_snap = {"stocks": evaluated, "settings": asdict(settings)}
        lists = {h: select_market_best(temp_snap, h, n=5) for h in ["short", "mid", "long"]}
        selected = list(dict.fromkeys(
            str(s.get("ticker")) for h in ["short", "mid", "long"] for s in lists.get(h, []) if s.get("ticker")
        ))
        missing = [t for t in selected if not _has_useful_research(next((s for s in evaluated if s.get("ticker") == t), {}).get("research"))]
        if not missing:
            break
        start_v = min(0.975, 0.955 + pass_idx * 0.008)
        _enrich_tickers(evaluated, data_dir, missing, progress=progress, progress_value=start_v, progress_end=min(0.985, start_v + 0.007))
        touched = set(missing)
        for stock in evaluated:
            if stock.get("ticker") in touched:
                apply_scores(stock)
                refresh_metadata(stock)

    # Capture the actual recommendation evidence coverage so the UI can detect
    # a degraded research provider instead of silently showing dashes.
    final_lists = {h: select_market_best({"stocks": evaluated, "settings": asdict(settings)}, h, n=5) for h in ["short", "mid", "long"]}
    final_selected = list({
        str(s.get("ticker")) for h in ["short", "mid", "long"]
        for s in final_lists.get(h, []) if s.get("ticker")
    })
    selected_objs = [s for s in evaluated if str(s.get("ticker")) in set(final_selected)]
    research_cov = {
        "selected": len(final_selected),
        "revenue": sum(1 for s in selected_objs if _research_availability(s.get("research"))["revenue"]),
        "financials": sum(1 for s in selected_objs if _research_availability(s.get("research"))["financials"]),
        "valuation": sum(1 for s in selected_objs if _research_availability(s.get("research"))["valuation"]),
        "institutional_flow": sum(1 for s in selected_objs if _research_availability(s.get("research"))["institutional_flow"]),
    }

    if progress:
        progress("完成", 1.0)
    dates = [s["price_date"] for s in evaluated if s.get("price_date")]
    latest_date = max(dates) if dates else str(market.get("price_date") or "")
    snap = {
        "snapshot_id": f"snap_{_taipei_timestamp().strftime('%Y%m%d_%H%M%S')}",
        "price_date": latest_date,
        "market": market,
        "coverage": {
            "requested": len(universe),
            "downloaded": int(dl.get("downloaded_tickers", 0)),
            "history_refresh_mode": dl.get("mode", ""),
            "history_refresh_requested": int(dl.get("requested", 0) or 0),
            "history_cache_covered_ratio": dl.get("covered_ratio"),
            "feature_valid": valid_count,
            "liquid": len(liquid),
            "stale_excluded": stale_count,
            "expected_date": expected_date,
            "calendar_expected_date": calendar_expected_date,
            "errors": dl.get("errors", []) + official_eod.get("errors", []),
            "official_eod": official_eod,
            "recommendation_research": research_cov,
        },
        "candidate_n": len(evaluated),
        "stocks": evaluated,
        "settings": asdict(settings),
        "source_type": "market_sector_stock_hierarchy",
        "sector_leadership": dict(sorted(sector_map.items(), key=lambda kv: float((kv[1] or {}).get("score") or 0.0), reverse=True)),
        "theme_leadership": dict(sorted(theme_map.items(), key=lambda kv: float((kv[1] or {}).get("score") or 0.0), reverse=True)),
        "model_version": OPERATIONS_VERSION,
        "generated_at": _taipei_timestamp().isoformat(timespec="seconds"),
        "charts": {},
    }
    try:
        atomic_write_json(data_dir / "dashboard_snapshot.json", compact_session_dashboard(snap))
    except Exception as exc:
        snap.setdefault("coverage", {}).setdefault("errors", []).append(f"snapshot write: {type(exc).__name__}: {exc}")
    return snap



def _horizon_selection_bonus(stock: dict, horizon: str) -> float:
    """Selection-time horizon identity bonus.

    Raw ranking scores remain the source of truth.  This function only helps the
    displayed shortlist better express what *short / mid / long* are supposed to
    mean, so the same stock does not flood all three views unless it is clearly
    compelling across horizons.
    """
    try:
        horizons = stock.get("horizons") or {}
        scores = {h: float((horizons.get(h) or {}).get("ranking_score") or 0.0) for h in ["short", "mid", "long"]}
        raw = scores.get(horizon, 0.0)
        others = [v for h, v in scores.items() if h != horizon]
        bonus = 0.0

        # Encourage each view to show stocks whose role best matches that horizon.
        role = str(((stock.get("role") or {}).get("code") or "")).strip()
        target_role = {"short": "tactical", "mid": "growth", "long": "core"}.get(horizon, "")
        if role and target_role:
            role_bonus = {
                "short": {"tactical": 3.2, "growth": 0.6, "core": -2.2},
                "mid": {"growth": 3.0, "core": 0.8, "tactical": -1.6},
                "long": {"core": 3.2, "growth": 0.5, "tactical": -3.0},
            }.get(horizon, {})
            bonus += float(role_bonus.get(role, 0.0))

        # Prefer the horizon where the stock has a clearer relative edge.
        if others:
            specialization = raw - max(others)
            bonus += float(np.clip(specialization * 0.35, -1.8, 1.8))

        stage = int(((stock.get("stage") or {}).get("code") or 0))
        setup = str(stock.get("setup") or "")
        block = horizons.get(horizon) or {}
        comp = block.get("score_components") or {}
        feat = ((block.get("forecast") or {}).get("features") or {})
        vol_ratio = float(feat.get("vol_ratio") or 1.0)
        gap20 = float(feat.get("gap20") or 0.0)
        atr_pct = float(feat.get("atr_pct") or 0.03)
        fund = float(comp.get("fundamental") or 0.0) if comp.get("fundamental") is not None else None
        valuation = float(comp.get("valuation_score") or 0.0) if comp.get("valuation_score") is not None else None
        liquidity = float(comp.get("liquidity") or 0.0)

        if horizon == "short":
            if setup in {"BREAKOUT", "WASHOUT", "RECLAIM"}:
                bonus += 1.2
            if stage in {2, 3}:
                bonus += 0.8
            if vol_ratio >= 1.15:
                bonus += 0.5
            if gap20 >= 0.16 or stage in {4, 5}:
                bonus -= 1.8
        elif horizon == "mid":
            if stage in {1, 2, 3}:
                bonus += 1.0
            if setup == "BREAKOUT":
                bonus += 0.4
            if fund is not None and fund >= 60:
                bonus += 0.6
            if stage in {4, 5}:
                bonus -= 1.0
        elif horizon == "long":
            if fund is not None and fund >= 65:
                bonus += 1.4
            if valuation is not None and valuation >= 55:
                bonus += 0.7
            if liquidity >= 40:
                bonus += 0.5
            if atr_pct <= 0.03:
                bonus += 0.5
            if stage in {4, 5}:
                bonus -= 1.4

        # If another horizon is much more suitable, apply a small display penalty.
        dominant = max(scores, key=scores.get) if scores else horizon
        if dominant != horizon and raw <= scores.get(dominant, raw) - 1.0:
            bonus -= 1.2
        return round(float(bonus), 2)
    except Exception:
        return 0.0

def _selection_adjusted_score(stock: dict, horizon: str, industry_counts: dict[str, int] | None = None) -> float:
    """Recommendation selection score; raw model score remains untouched."""
    block = stock.get("horizons", {}).get(horizon, {}) or {}
    raw = float(block.get("ranking_score", -1) or -1)
    fit = _horizon_selection_bonus(stock, horizon)
    industry = str(stock.get("fine_industry") or stock.get("industry") or "未分類")
    repeats = (industry_counts or {}).get(industry, 0)
    sector_penalty = {"short": 0.8, "mid": 1.3, "long": 1.8}.get(horizon, 1.0) * repeats

    availability = _research_availability(stock.get("research") or {})
    company_evidence = int(availability["revenue"]) + int(availability["financials"]) + int(availability["valuation"])
    evidence_penalty = 0.0
    if company_evidence == 0:
        evidence_penalty += {"short": 2.5, "mid": 7.0, "long": 12.0}[horizon]
    elif company_evidence == 1:
        evidence_penalty += {"short": 0.5, "mid": 2.0, "long": 4.0}[horizon]

    state = str(block.get("entry_state") or "")
    timing = 0.0
    if horizon == "short":
        if state == "CONDITIONS_MET_NOT_FILLED":
            timing += 1.2
        elif state == "DO_NOT_CHASE":
            timing -= 4.0
        elif state in {"WAIT_ENTRY_ZONE", "WAIT_BREAKOUT", "WAIT_CONFIRMATION"}:
            timing -= 1.0
        stage = int(((stock.get("stage") or {}).get("code") or 0))
        if stage == 0:
            timing -= 0.8

    return raw + fit + timing - sector_penalty - evidence_penalty


def select_cross_horizon_shortlists(
    snap: dict | None,
    n: int = 5,
    max_appearances: int = 2,
) -> dict[str, list]:
    """Joint shortlists without forcing a stock into a weak 'primary' horizon.

    V14.0 over-corrected overlap by assigning every stock to exactly one primary
    horizon first. That could push a Core/整理 name into the short list simply
    because its *relative* short rank was its least-bad rank. V14.1 instead fills
    each horizon from its own suitability score and uses only a soft overlap cost.
    """
    out = {"short": [], "mid": [], "long": []}
    if not snap or not isinstance(snap, dict):
        return out
    stocks = [s for s in snap.get("stocks", []) if isinstance(s, dict)]
    if not stocks:
        return out

    horizons = ["short", "mid", "long"]
    appearances: dict[str, int] = {}
    sector_counts: dict[str, dict[str, int]] = {h: {} for h in horizons}
    model_family = str((snap.get("settings") or {}).get("model_family") or "full")
    needs_business = model_family in {"business_confirmed", "full"}

    # Fill one rank at a time across horizons. A second appearance costs points;
    # triple overlap is forbidden. No low-quality replacement is forced merely
    # to manufacture visual variety.
    for _rank in range(n):
        for h in horizons:
            candidates = []
            for s in stocks:
                ticker = str(s.get("ticker") or "")
                if not ticker or s in out[h] or appearances.get(ticker, 0) >= max_appearances:
                    continue
                raw_h = (s.get("horizons", {}).get(h, {}) or {}).get("ranking_score")
                if raw_h is None or float(raw_h) < 0:
                    continue

                # Production evidence gate. Unattempted research may enter the
                # provisional shortlist and will be fetched by the final-enrichment
                # loop. Once a fetch was attempted, however, a mid/long idea must
                # have enough company evidence to remain in those horizons.
                research = s.get("research") or {}
                attempted = bool(research.get("fetched_at"))
                if needs_business and attempted:
                    av = _research_availability(research)
                    if h == "mid" and not (av["revenue"] or av["financials"]):
                        continue
                    if h == "long" and not (av["financials"] and (av["revenue"] or av["valuation"])):
                        continue

                base = _selection_adjusted_score(s, h, sector_counts[h])
                overlap_penalty = 3.5 * appearances.get(ticker, 0)
                candidates.append((base - overlap_penalty, s))
            if not candidates:
                continue
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, pick = candidates[0]
            out[h].append(pick)
            ticker = str(pick.get("ticker") or "")
            appearances[ticker] = appearances.get(ticker, 0) + 1
            industry = str(pick.get("fine_industry") or pick.get("industry") or "未分類")
            sector_counts[h][industry] = sector_counts[h].get(industry, 0) + 1

    for h in horizons:
        out[h].sort(key=lambda s: _selection_adjusted_score(s, h), reverse=True)
        out[h] = out[h][:n]
    return out


def shortlist_overlap_metrics(shortlists: dict[str, list]) -> dict:
    """Small diagnostic used by UI/tests to catch horizon-collapse regressions."""
    sets = {h: {str(s.get("ticker")) for s in shortlists.get(h, [])} for h in ["short", "mid", "long"]}
    union = set().union(*sets.values()) if sets else set()
    total_slots = sum(len(v) for v in sets.values())
    duplicates = total_slots - len(union)
    triple = sets["short"] & sets["mid"] & sets["long"]
    return {
        "total_slots": total_slots,
        "unique_tickers": len(union),
        "duplicate_slots": duplicates,
        "triple_overlap": sorted(triple),
        "overlap_ratio": round(duplicates / max(1, total_slots), 3),
    }


def select_market_best(snap: dict | None, horizon: str, n: int = 5) -> list:
    """True horizon ranking, independent of the other horizon lists.

    V16 deliberately separates *signal* from *portfolio allocation*. If the same
    stock is genuinely strong in all three horizons it is allowed to appear in
    all three signal lists and is labelled a cross-horizon consensus. De-dup is
    applied only by `select_cross_horizon_shortlists` when a user wants an actual
    diversified allocation. This avoids hiding a real multi-horizon leader merely
    to make the UI look varied.
    """
    if horizon not in {"short", "mid", "long"} or not snap or not isinstance(snap, dict):
        return []
    stocks = [s for s in snap.get("stocks", []) if isinstance(s, dict)]
    if not stocks:
        return []
    model_family = str((snap.get("settings") or {}).get("model_family") or "full")
    needs_business = model_family in {"business_confirmed", "full"}
    pool = []
    for s in stocks:
        block = (s.get("horizons", {}).get(horizon, {}) or {})
        raw_h = block.get("ranking_score")
        if raw_h is None:
            continue
        research = s.get("research") or {}
        attempted = bool(research.get("fetched_at"))
        if needs_business and attempted:
            av = _research_availability(research)
            if horizon == "mid" and not (av["revenue"] or av["financials"]):
                continue
            if horizon == "long" and not (av["financials"] and (av["revenue"] or av["valuation"])):
                continue
        pool.append(s)

    chosen: list[dict] = []
    counts: dict[str, int] = {}
    remaining = list(pool)
    while remaining and len(chosen) < n:
        scored = []
        for s in remaining:
            score = _selection_adjusted_score(s, horizon, counts)
            scored.append((score, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        _, pick = scored[0]
        chosen.append(pick)
        remaining.remove(pick)
        industry = str(pick.get("fine_industry") or pick.get("industry") or "未分類")
        counts[industry] = counts.get(industry, 0) + 1
    return chosen[:n]

def select_view(snap: dict | None, horizon: str, qualified: bool = True, n: int = 5, **kwargs) -> list:
    return select_market_best(snap, horizon, n)


def _evaluate_one(ticker: str, name: str, industry: str, df: pd.DataFrame, settings: RunSettings, market_ret20: float, research: dict, benchmark_df: pd.DataFrame | None = None) -> dict:
    price = float(df["Close"].iloc[-1])
    stock = {
        "ticker": ticker,
        "name": name,
        "industry": industry,
        "fine_industry": fine_industry(ticker, name, industry),
        "theme": theme_bucket(ticker, name, industry, fine_industry(ticker, name, industry)),
        "price": price,
        "price_date": str(df.index[-1].date()),
        "avg_volume_20d": float(df["Volume"].tail(20).mean()),
        "avg_turnover_20d": float((df["Close"].tail(20) * df["Volume"].tail(20)).mean()),
        "sector_strength": {"score": 50.0, "status": "個股診斷"},
        "theme_strength": {"score": 50.0, "status": "個股診斷", "name": theme_bucket(ticker, name, industry, fine_industry(ticker, name, industry))},
        "setup": setup_from_df(df),
        "horizons": {},
        "research": research or {},
    }
    forecasts = estimate_all_horizons(df, settings, market_ret20, benchmark_df=benchmark_df)
    for h in ["short", "mid", "long"]:
        plan = generate_trade_plan(df, h)
        est = forecasts[h]
        stock["horizons"][h] = {
            "plan": plan,
            "entry_state": evaluate_entry_state(df, plan),
            "forecast": est,
            "qualification": {"research_qualified": bool(est.get("composite_factor_score", 0) >= 35), "rank_first": True},
        }
        _combined_ranking_score(stock, h, settings.model_family)
    _decorate_role_stage(stock)
    return stock


def diagnose(code: str, snap: dict | None, data_dir: Path) -> dict:
    clean = str(code or "").strip()
    if not clean:
        return {"snapshot_id": "snap_none", "error": "EMPTY_CODE"}
    snap_id = snap.get("snapshot_id", "snap_unknown") if isinstance(snap, dict) else "snap_none"
    for s in (snap.get("stocks", []) if isinstance(snap, dict) else []):
        if isinstance(s, dict) and (s.get("ticker") == clean or str(s.get("ticker", "")).startswith(clean + ".")):
            return {"snapshot_id": snap_id, "stock": s}

    ticker_options = [clean] if "." in clean else [f"{clean}.TW", f"{clean}.TWO"]
    store = DailyPriceStore(Path(data_dir) / "daily_prices.sqlite")
    store.batch_fetch_and_update(ticker_options, period="5y")
    diag_cal = calendar_reference(Path(data_dir), now=_taipei_timestamp())
    store.overlay_official_eod(target_date=str(diag_cal.get("expected_date") or ""))
    ticker = None
    df = pd.DataFrame()
    for t in ticker_options:
        trial = store.get_prices(t)
        if not trial.empty:
            ticker, df = t, trial
            break
    if ticker is None or df.empty:
        return {"snapshot_id": snap_id, "error": "PRICE_DATA_UNAVAILABLE", "code": clean}

    settings_dict = snap.get("settings", {}) if isinstance(snap, dict) else {}
    try:
        allowed = set(RunSettings.__dataclass_fields__)
        settings = RunSettings(**{k: v for k, v in settings_dict.items() if k in allowed})
    except Exception:
        settings = RunSettings()
    market_ret20 = float((snap.get("market", {}) if isinstance(snap, dict) else {}).get("ret20") or 0.0)
    benchmark_df = store.get_prices("^TWII")
    research = ResearchDataClient(Path(data_dir) / "research_cache.sqlite").stock_research(
        ticker, include_branch=(os.getenv("ENABLE_BRANCH_FLOW", "0") == "1")
    )
    # Recover the public company name / broad industry when possible so a direct
    # diagnosis gets the same fine-industry label as a market scan.
    display_name, broad_industry = clean, "個股診斷"
    try:
        universe = fetch_twse_universe()
        code = ticker.split(".")[0]
        hit = universe[universe["ticker"].astype(str).str.startswith(code + ".")]
        if not hit.empty:
            display_name = str(hit.iloc[0].get("name") or clean)
            broad_industry = str(hit.iloc[0].get("industry") or "個股診斷")
    except Exception:
        pass
    stock = _evaluate_one(ticker, display_name, broad_industry, df, settings, market_ret20, research, benchmark_df=benchmark_df)
    try:
        sector_map = (snap or {}).get("sector_leadership") or {} if isinstance(snap, dict) else {}
        theme_map = (snap or {}).get("theme_leadership") or {} if isinstance(snap, dict) else {}
        if broad_industry in sector_map:
            stock["sector_strength"] = sector_map[broad_industry]
        th = str(stock.get("theme") or "")
        if th in theme_map:
            stock["theme_strength"] = {**theme_map[th], "name": th}
            for h in ["short", "mid", "long"]:
                _combined_ranking_score(stock, h, settings.model_family)
            _decorate_role_stage(stock)
    except Exception:
        pass
    return {"snapshot_id": snap_id, "stock": stock}
