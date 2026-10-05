"""Intraday overlay for Taiwan Alpha Radar V16.1.

The daily model decides *what is worth watching*. This module answers
*what is happening now* and re-orders that already-vetted pool. It never
replaces the long-horizon evidence with a partial intraday candle.

Realtime provider chain:
1) FinMind realtime snapshot when a token is configured;
2) TWSE/TPEx Market Information System (MIS) 5-second public quote page as a
   no-token fallback for the user's own research. Public redistribution of TWSE
   realtime data can require a separate information-use agreement.
"""
from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import requests

REALTIME_URL = "https://api.finmindtrade.com/api/v4/taiwan_stock_tick_snapshot"
MIS_URL = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
MIS_HOME = "https://mis.twse.com.tw/stock/index.jsp"
INTRADAY_BLEND = {"short": 0.45, "mid": 0.15, "long": 0.05}


def _taipei_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))


def _stock_id(ticker: str) -> str:
    return str(ticker or "").split(".")[0].strip()


def market_is_open(now: datetime.datetime | None = None) -> bool:
    now = now or _taipei_now()
    if now.weekday() >= 5:
        return False
    t = now.timetz().replace(tzinfo=None)
    return datetime.time(9, 0) <= t <= datetime.time(13, 30)


def _finite(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _clip01(v: float, lo: float, hi: float) -> float:
    if not np.isfinite(v):
        return 0.5
    if hi <= lo:
        return 0.5
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


@dataclass
class RealtimeStatus:
    available: bool
    reason: str = ""
    quote_date: str = ""
    source: str = "FinMind realtime snapshot"
    market_open: bool = False
    fetched_at: str = ""


class FinMindRealtimeClient:
    """Realtime client with a no-token official MIS fallback.

    The class name is kept for backward compatibility with app.py.  When a
    FinMind token exists it is tried first.  If the token is absent, lacks the
    realtime entitlement, or the response is stale/unavailable, the client
    falls back to TWSE/TPEx MIS quotes.
    """
    def __init__(self, token: str | None = None):
        self.token = (token if token is not None else os.getenv("FINMIND_TOKEN", "")).strip()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 AlphaRadar/16.1",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://mis.twse.com.tw/stock/index.jsp",
        })
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        self._mis_bootstrapped = False

    @property
    def configured(self) -> bool:
        # MIS does not require a user token, so the realtime layer is configured
        # even when FINMIND_TOKEN is absent.
        return True

    def _request_ids(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        params: list[tuple[str, str]] = []
        for stock_id in ids:
            params.append(("data_id", stock_id))
        if self.token:
            params.append(("token", self.token))
        r = self.session.get(REALTIME_URL, params=params, timeout=10)
        r.raise_for_status()
        js = r.json()
        if not isinstance(js, dict):
            return []
        status = js.get("status")
        if status not in (None, 200, "200"):
            msg = str(js.get("msg") or js.get("message") or "FINMIND_API_ERROR")
            raise RuntimeError(msg)
        data = js.get("data", [])
        return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []

    @staticmethod
    def _num(v, default=np.nan):
        try:
            if v in (None, "", "-"):
                return default
            x = float(str(v).replace(",", ""))
            return x if np.isfinite(x) else default
        except Exception:
            return default

    @classmethod
    def _first_book_value(cls, v):
        if v in (None, "", "-"):
            return np.nan
        for part in str(v).split("_"):
            x = cls._num(part)
            if np.isfinite(x):
                return x
        return np.nan

    def _bootstrap_mis(self):
        if self._mis_bootstrapped:
            return
        try:
            self.session.get(MIS_HOME, timeout=5)
        except Exception:
            pass
        self._mis_bootstrapped = True

    @staticmethod
    def _mis_channel(ticker: str) -> str:
        ticker = str(ticker or "")
        stock_id = _stock_id(ticker)
        prefix = "otc" if ticker.endswith(".TWO") else "tse"
        return f"{prefix}_{stock_id}.tw"

    def _mis_convert(self, raw: dict) -> dict | None:
        if not isinstance(raw, dict):
            return None
        ch = str(raw.get("ch") or "")
        code = str(raw.get("c") or "").strip()
        if "t00" in ch:
            stock_id = "001"
        elif "o00" in ch:
            stock_id = "101"
        else:
            stock_id = code
        if not stock_id:
            return None

        prev = self._num(raw.get("y"))
        close = self._num(raw.get("z"))
        if not np.isfinite(close):
            close = self._num(raw.get("pz"))
        if not np.isfinite(close):
            bid = self._first_book_value(raw.get("b"))
            ask = self._first_book_value(raw.get("a"))
            if np.isfinite(bid) and np.isfinite(ask):
                close = (bid + ask) / 2.0
            elif np.isfinite(prev):
                close = prev

        change_rate = ((close / prev) - 1.0) * 100.0 if np.isfinite(close) and np.isfinite(prev) and prev > 0 else 0.0
        volume_lots = self._num(raw.get("v"), 0.0)
        # MIS cumulative volume is published in round lots for equities.  The
        # historical cache uses shares, so normalize to shares for comparisons.
        total_volume = volume_lots * 1000.0 if stock_id not in {"001", "101"} else volume_lots
        amount = close * total_volume if np.isfinite(close) and total_volume > 0 and stock_id not in {"001", "101"} else np.nan

        d = str(raw.get("d") or "")
        t = str(raw.get("t") or "")
        date_text = ""
        if len(d) == 8 and d.isdigit():
            date_text = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            if t and t != "-":
                date_text += f" {t}"

        return {
            "stock_id": stock_id,
            "date": date_text,
            "close": None if not np.isfinite(close) else float(close),
            "open": self._num(raw.get("o")),
            "high": self._num(raw.get("h")),
            "low": self._num(raw.get("l")),
            "average_price": np.nan,  # MIS does not expose VWAP in this payload.
            "change_rate": round(float(change_rate), 4),
            "total_volume": float(total_volume),
            "total_amount": None if not np.isfinite(amount) else float(amount),
            "buy_price": self._first_book_value(raw.get("b")),
            "sell_price": self._first_book_value(raw.get("a")),
            "buy_volume": self._first_book_value(raw.get("g")),
            "sell_volume": self._first_book_value(raw.get("f")),
            "source": "TWSE/TPEx MIS",
        }

    def _mis_snapshots(self, tickers: list[str], batch_size: int = 28) -> tuple[pd.DataFrame, RealtimeStatus]:
        self._bootstrap_mis()
        channels = [self._mis_channel(t) for t in tickers if _stock_id(t)]
        # Market indices are required for relative-strength scoring.
        channels += ["tse_t00.tw", "otc_o00.tw"]
        channels = list(dict.fromkeys(channels))
        rows: list[dict] = []
        try:
            for i in range(0, len(channels), batch_size):
                batch = channels[i:i + batch_size]
                params = {
                    "ex_ch": "|".join(batch),
                    "json": "1",
                    "delay": "0",
                    "_": str(int(_taipei_now().timestamp() * 1000)),
                }
                r = self.session.get(MIS_URL, params=params, timeout=10)
                r.raise_for_status()
                js = r.json()
                if not isinstance(js, dict) or str(js.get("rtcode", "0000")) not in {"0000", "0"}:
                    continue
                for raw in js.get("msgArray", []) or []:
                    item = self._mis_convert(raw)
                    if item:
                        rows.append(item)
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", "")
            return pd.DataFrame(), RealtimeStatus(False, f"MIS_HTTP_{status or 'ERROR'}", source="TWSE/TPEx MIS", market_open=market_is_open())
        except Exception as exc:
            return pd.DataFrame(), RealtimeStatus(False, f"MIS_{type(exc).__name__}", source="TWSE/TPEx MIS", market_open=market_is_open())

        if not rows:
            return pd.DataFrame(), RealtimeStatus(False, "MIS_NO_ROWS", source="TWSE/TPEx MIS", market_open=market_is_open())
        df = pd.DataFrame(rows)
        ds = pd.to_datetime(df.get("date"), errors="coerce") if "date" in df.columns else pd.Series(dtype="datetime64[ns]")
        quote_date = str(ds.max().date()) if len(ds) and ds.notna().any() else ""
        now = _taipei_now()
        if market_is_open(now) and quote_date and quote_date != now.date().isoformat():
            return pd.DataFrame(), RealtimeStatus(False, f"MIS_STALE_{quote_date}", quote_date=quote_date, source="TWSE/TPEx MIS", market_open=True, fetched_at=now.isoformat(timespec="seconds"))
        return df.drop_duplicates("stock_id", keep="last"), RealtimeStatus(
            True, quote_date=quote_date, source="TWSE/TPEx MIS", market_open=market_is_open(now), fetched_at=now.isoformat(timespec="seconds")
        )

    def _finmind_snapshots(self, tickers: list[str], batch_size: int = 45) -> tuple[pd.DataFrame, RealtimeStatus]:
        if not self.token:
            return pd.DataFrame(), RealtimeStatus(False, "FINMIND_TOKEN_NOT_CONFIGURED", source="FinMind realtime snapshot", market_open=market_is_open())
        ids = list(dict.fromkeys([_stock_id(t) for t in tickers if _stock_id(t)]))
        ids_with_index = ids + ["001", "101"]
        rows: list[dict] = []
        try:
            for i in range(0, len(ids_with_index), batch_size):
                rows.extend(self._request_ids(ids_with_index[i:i + batch_size]))
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", "")
            return pd.DataFrame(), RealtimeStatus(False, f"HTTP_{status or 'ERROR'}", source="FinMind realtime snapshot", market_open=market_is_open())
        except Exception as exc:
            return pd.DataFrame(), RealtimeStatus(False, f"{type(exc).__name__}", source="FinMind realtime snapshot", market_open=market_is_open())
        if not rows:
            return pd.DataFrame(), RealtimeStatus(False, "NO_REALTIME_ROWS", source="FinMind realtime snapshot", market_open=market_is_open())
        df = pd.DataFrame(rows)
        if "stock_id" not in df.columns:
            return pd.DataFrame(), RealtimeStatus(False, "MISSING_STOCK_ID", source="FinMind realtime snapshot", market_open=market_is_open())
        df["stock_id"] = df["stock_id"].astype(str)
        ds = pd.to_datetime(df["date"], errors="coerce") if "date" in df.columns else pd.Series(dtype="datetime64[ns]")
        quote_date = str(ds.max().date()) if len(ds) and ds.notna().any() else ""
        now = _taipei_now()
        if market_is_open(now) and quote_date and quote_date != now.date().isoformat():
            return pd.DataFrame(), RealtimeStatus(False, f"STALE_REALTIME_{quote_date}", quote_date=quote_date, source="FinMind realtime snapshot", market_open=True, fetched_at=now.isoformat(timespec="seconds"))
        return df.drop_duplicates("stock_id", keep="last"), RealtimeStatus(True, quote_date=quote_date, source="FinMind realtime snapshot", market_open=market_is_open(now), fetched_at=now.isoformat(timespec="seconds"))

    def snapshots(self, tickers: Iterable[str], batch_size: int = 45) -> tuple[pd.DataFrame, RealtimeStatus]:
        tickers = list(dict.fromkeys([str(t) for t in tickers if _stock_id(t)]))
        # Prefer the richer FinMind snapshot if entitlement exists, but never let
        # a missing token/permission silently disable intraday ranking.
        if self.token:
            df, status = self._finmind_snapshots(tickers, batch_size=batch_size)
            if status.available and not df.empty:
                return df, status
        return self._mis_snapshots(tickers)

def _quote_features(row: pd.Series, market_rate_pp: float) -> dict:
    close = _finite(row.get("close"))
    open_ = _finite(row.get("open"))
    high = _finite(row.get("high"))
    low = _finite(row.get("low"))
    avg = _finite(row.get("average_price"))
    change_pp = _finite(row.get("change_rate"), 0.0)
    vol_ratio = _finite(row.get("volume_ratio"), np.nan)
    amount = _finite(row.get("total_amount"), _finite(row.get("amount"), np.nan))
    buy_p = _finite(row.get("buy_price"))
    sell_p = _finite(row.get("sell_price"))
    buy_v = _finite(row.get("buy_volume"))
    sell_v = _finite(row.get("sell_volume"))

    relative_pp = change_pp - market_rate_pp
    vwap_gap_pp = ((close / avg) - 1.0) * 100.0 if np.isfinite(close) and np.isfinite(avg) and avg > 0 else np.nan
    open_move_pp = ((close / open_) - 1.0) * 100.0 if np.isfinite(close) and np.isfinite(open_) and open_ > 0 else np.nan
    day_pos = (close - low) / (high - low) if all(np.isfinite(x) for x in [close, high, low]) and high > low else 0.5
    mid = (buy_p + sell_p) / 2.0 if np.isfinite(buy_p) and np.isfinite(sell_p) and buy_p + sell_p > 0 else np.nan
    spread_pp = ((sell_p - buy_p) / mid) * 100.0 if np.isfinite(mid) and mid > 0 else np.nan
    bid_pressure = buy_v / (buy_v + sell_v) if np.isfinite(buy_v) and np.isfinite(sell_v) and buy_v + sell_v > 0 else 0.5

    # Deliberately bounded, interpretable factor map.  A hot print alone cannot
    # dominate the result; strength must be supported by relative performance,
    # volume and price holding above the intraday average.
    rel_s = _clip01(relative_pp, -2.0, 4.0) * 100
    vwap_s = _clip01(vwap_gap_pp, -1.5, 2.0) * 100
    vol_s = _clip01(vol_ratio, 0.55, 2.5) * 100 if np.isfinite(vol_ratio) else 50.0
    pos_s = _clip01(day_pos, 0.18, 0.88) * 100
    open_s = _clip01(open_move_pp, -2.5, 4.0) * 100
    bid_s = _clip01(bid_pressure, 0.35, 0.65) * 100
    if np.isfinite(amount) and amount > 0:
        # 30m to 2.5b TWD mapped logarithmically, preventing mega-caps from
        # winning solely because of absolute turnover.
        turn_s = _clip01(np.log10(amount), np.log10(30_000_000), np.log10(2_500_000_000)) * 100
    else:
        turn_s = 45.0

    raw = 0.30 * rel_s + 0.20 * vwap_s + 0.18 * vol_s + 0.12 * pos_s + 0.08 * open_s + 0.05 * bid_s + 0.07 * turn_s
    overheat_penalty = 0.0
    if change_pp > 6.0 and np.isfinite(vwap_gap_pp):
        overheat_penalty += min(14.0, max(0.0, (change_pp - 6.0) * 2.0 + max(0.0, vwap_gap_pp - 3.0) * 2.0))
    if day_pos > 0.96 and np.isfinite(vwap_gap_pp) and vwap_gap_pp > 3.0:
        overheat_penalty += 4.0
    spread_penalty = max(0.0, (spread_pp - 0.6) * 6.0) if np.isfinite(spread_pp) else 0.0
    score = float(np.clip(raw - overheat_penalty - spread_penalty, 0, 100))

    if overheat_penalty >= 8.0:
        state = "æ¼²å¤ªå¿«ï¼ä¸è¿½"
    elif (relative_pp <= -1.5) or (np.isfinite(vwap_gap_pp) and vwap_gap_pp < -1.2 and day_pos < 0.35):
        state = "è½å¼±ï¼åç¥é"
    elif score >= 70 and relative_pp >= 0.8 and (not np.isfinite(vol_ratio) or vol_ratio >= 1.05) and (not np.isfinite(vwap_gap_pp) or vwap_gap_pp >= 0):
        state = "è½å¼·ï¼å¯åªåç"
    elif score >= 60 and (not np.isfinite(vwap_gap_pp) or vwap_gap_pp >= -0.2):
        state = "èµ°å¢ç©©ï¼ç­æå"
    elif np.isfinite(vol_ratio) and vol_ratio >= 1.2 and score >= 50:
        state = "æéï¼éæ²çªç ´"
    else:
        state = "ç­ç¢ºèª"

    reasons: list[str] = []
    if relative_pp >= 1.0:
        reasons.append(f"æ¯å¤§ç¤å¼· {relative_pp:.1f} åç¾åé»")
    elif relative_pp <= -1.0:
        reasons.append(f"æ¯å¤§ç¤å¼± {abs(relative_pp):.1f} åç¾åé»")
    if np.isfinite(vol_ratio) and vol_ratio >= 1.25:
        reasons.append(f"éè½ç´å¹³å¸¸ {vol_ratio:.1f} å")
    if np.isfinite(vwap_gap_pp):
        if vwap_gap_pp >= 0.4:
            reasons.append("å®å¨ç¤ä¸­åå¹ä¸æ¹")
        elif vwap_gap_pp <= -0.5:
            reasons.append("è·å°ç¤ä¸­åå¹ä¸æ¹")
    if not reasons:
        reasons.append("ç¤ä¸­è¨èå°æªå½¢ææé¡¯åªå¢")

    return {
        "intraday_score": round(score, 1),
        "state": state,
        "change_rate_pct": round(change_pp, 2),
        "market_change_rate_pct": round(float(market_rate_pp), 2),
        "relative_market_pct_pt": round(float(relative_pp), 2),
        "vwap_gap_pct": None if not np.isfinite(vwap_gap_pp) else round(float(vwap_gap_pp), 2),
        "volume_ratio": None if not np.isfinite(vol_ratio) else round(float(vol_ratio), 2),
        "day_position": round(float(day_pos), 3),
        "spread_pct": None if not np.isfinite(spread_pp) else round(float(spread_pp), 3),
        "bid_pressure": round(float(bid_pressure), 3),
        "price": None if not np.isfinite(close) else float(close),
        "average_price": None if not np.isfinite(avg) else float(avg),
        "open": None if not np.isfinite(open_) else float(open_),
        "high": None if not np.isfinite(high) else float(high),
        "low": None if not np.isfinite(low) else float(low),
        "total_amount": None if not np.isfinite(amount) else float(amount),
        "overheat_penalty": round(float(overheat_penalty), 1),
        "reasons": reasons[:2],
    }


def rerank_snapshot(snap: dict, horizon: str, realtime_df: pd.DataFrame, top_n: int = 10) -> list[dict]:
    if not snap or realtime_df is None or realtime_df.empty:
        return []
    if horizon not in INTRADAY_BLEND:
        horizon = "short"
    rows = {str(r["stock_id"]): r for _, r in realtime_df.iterrows()}
    twse = rows.get("001")
    otc = rows.get("101")
    out: list[dict] = []
    for stock in snap.get("stocks", []):
        ticker = str(stock.get("ticker", ""))
        q = rows.get(_stock_id(ticker))
        if q is None:
            continue
        if market_is_open() and "date" in q:
            qdt = pd.to_datetime(q.get("date"), errors="coerce")
            if pd.notna(qdt) and str(qdt.date()) != _taipei_now().date().isoformat():
                continue
        index_row = otc if ticker.endswith(".TWO") else twse
        market_rate = _finite(index_row.get("change_rate"), 0.0) if index_row is not None else 0.0
        q = q.copy()
        # MIS gives cumulative volume but not a ready-made volume ratio. Convert
        # current pace into an estimated full-session ratio against the stock's
        # 20-day average volume. The floor prevents the first few minutes from
        # exploding the estimate.
        if not np.isfinite(_finite(q.get("volume_ratio"))):
            current_volume = _finite(q.get("total_volume"), np.nan)
            avg_volume = _finite(stock.get("avg_volume_20d"), np.nan)
            now = _taipei_now()
            mins = max(0.0, (now.hour * 60 + now.minute + now.second / 60.0) - 9 * 60)
            session_fraction = float(np.clip(mins / 270.0, 0.10, 1.0))
            if np.isfinite(current_volume) and np.isfinite(avg_volume) and avg_volume > 0:
                q["volume_ratio"] = float(np.clip((current_volume / avg_volume) / session_fraction, 0.05, 6.0))
        intraday = _quote_features(q, market_rate)
        daily_score = _finite(stock.get("horizons", {}).get(horizon, {}).get("ranking_score"), np.nan)
        if not np.isfinite(daily_score):
            continue
        w = INTRADAY_BLEND[horizon]
        final_score = float(np.clip(daily_score * (1.0 - w) + intraday["intraday_score"] * w, 0, 100))
        item = dict(stock)
        item["intraday"] = intraday
        item["daily_ranking_score"] = round(float(daily_score), 1)
        item["live_ranking_score"] = round(final_score, 1)
        out.append(item)
    out.sort(key=lambda x: x.get("live_ranking_score", -1), reverse=True)
    return out[: max(1, int(top_n))]


def candidate_tickers(snap: dict, per_horizon: int = 40) -> list[str]:
    """Union of the daily model's front-ranked stocks, bounded for realtime calls."""
    if not snap:
        return []
    stocks = [x for x in snap.get("stocks", []) if isinstance(x, dict)]
    chosen: list[str] = []
    for h in ["short", "mid", "long"]:
        ranked = sorted(stocks, key=lambda x: x.get("horizons", {}).get(h, {}).get("ranking_score", -1), reverse=True)
        chosen.extend([str(x.get("ticker")) for x in ranked[:per_horizon] if x.get("ticker")])
    return list(dict.fromkeys(chosen))[:100]
