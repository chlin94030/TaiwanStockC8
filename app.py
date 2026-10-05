"""Alpha Radar V16.1 — final evidence + automatic intraday overlay UI."""
from __future__ import annotations

from pathlib import Path
import gc
import html
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import radar_service as service
from industry_profile import fine_industry
from intraday_engine import FinMindRealtimeClient, candidate_tickers, market_is_open, rerank_snapshot
from market_data import DailyPriceStore, _taipei_timestamp, provider_runtime_status
from policy_engine import HORIZONS
from presentation import investment_view, intraday_view
from return_first_model import holding_review
from trading_calendar import calendar_reference
from operational_tools import ScanBusyError, safe_error_text, scan_lock

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("ALPHA_RADAR_DATA_DIR", str(ROOT / "data")))

VIEWS = ["總覽", "即時", "短線", "中線", "長線", "個股"]
H_LABEL = {"short": "短線 · 約2週", "mid": "中線 · 約2個月", "long": "長線 · 約6個月"}
REGIME = {"BULL": "大盤偏多", "NEUTRAL": "大盤整理", "BEAR": "大盤偏弱", "UNKNOWN": "大盤待確認"}
FAMILY_LABELS = {
    "完整資料": "full",
    "價格＋公司資料": "business_confirmed",
    "價格＋法人": "flow_confirmed",
    "只看價格": "price_only",
}
HOLD_LABELS = {
    "DATA_UNVERIFIED": "價格尚未核對。",
    "ORIGINAL_STRUCTURE_INVALIDATED": "已跌破原防守價，需優先重新評估。",
    "ORIGINAL_THESIS_INVALIDATED": "原買進理由已不成立，需重新評估。",
    "PROTECTION_TRIGGER_REVIEW_EXECUTION": "已碰到獲利保護價，請依原紀律處理。",
    "ORIGINAL_THESIS_UNKNOWN_MANUAL_REVIEW": "缺少原始防守條件，無法自動判斷。",
    "ORIGINAL_RULES_NOT_BREACHED_NOT_A_RETURN_GUARANTEE": "原防守條件尚未被破壞。",
}

def _bootstrap_secrets():
    """Expose Streamlit secrets to the data layer without hard-coding tokens."""
    for key in ("FINMIND_TOKEN",):
        if os.getenv(key):
            continue
        try:
            value = st.secrets.get(key)
        except Exception:
            value = None
        if value not in (None, ""):
            os.environ[key] = str(value)


CSS = r"""
<style>
:root{
  --paper:#f5f4ef; --surface:#fffefb; --ink:#111318; --muted:#6b6f78;
  --line:#d9d8d2; --blue:#0b57d0; --red:#d92d20; --green:#07875d;
  --amber:#b35c00; --soft:#ecebe5;
}
.stApp{background:var(--paper);color:var(--ink);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
header[data-testid="stHeader"]{background:rgba(245,244,239,.96);backdrop-filter:blur(8px);}
.block-container{max-width:1120px;padding-top:4.75rem;padding-bottom:4rem;}
@supports (padding-top:env(safe-area-inset-top)){
 .block-container{padding-top:calc(4.75rem + env(safe-area-inset-top));}
}
#MainMenu,footer{visibility:hidden;}
.mast{display:grid;grid-template-columns:1fr auto;align-items:end;border-top:5px solid var(--ink);border-bottom:1px solid var(--ink);padding:13px 1px 11px;margin-bottom:10px;}
.brand{font-size:1.62rem;font-weight:950;letter-spacing:-.055em;line-height:1}.brand span{color:var(--blue)}
.version{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.76rem;color:var(--muted);letter-spacing:.06em;}
.marketbar{display:flex;gap:13px;flex-wrap:wrap;align-items:center;padding:7px 0 10px;border-bottom:1px solid var(--line);font-size:.88rem;color:var(--muted);}
.marketbar b{color:var(--ink)}
.section-kicker{font-size:.72rem;font-weight:900;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin:18px 0 5px;}
.section-title{font-size:1.32rem;font-weight:950;letter-spacing:-.025em;margin:0 0 9px;}
.rankrow{display:grid;grid-template-columns:44px minmax(0,1fr) auto;gap:10px;align-items:center;background:var(--surface);border-top:1px solid var(--ink);border-bottom:1px solid var(--line);padding:13px 10px 12px;margin-top:9px;}
.rankno{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:1.28rem;font-weight:900;color:#a3a39d;align-self:start;padding-top:2px;}
.stockname{font-size:1.28rem;font-weight:950;line-height:1.05;letter-spacing:-.025em}.stockcode{font-size:.79rem;font-weight:750;color:var(--muted);margin-left:5px;}
.industry{font-size:.82rem;color:#4d5159;margin-top:4px;line-height:1.15;}.tagrow{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}.tag{display:inline-flex;align-items:center;padding:2px 8px;border-radius:999px;border:1px solid var(--line);background:#f7f6f1;font-size:.73rem;font-weight:850;color:#434851}.tag.blue{color:#0b57d0;border-color:#c9daf8;background:#f2f7ff}.tag.red{color:#8a3b11;border-color:#ead2c6;background:#fff6f1}
.quote{text-align:right;min-width:92px}.price{font-size:1.30rem;font-weight:950;line-height:1}.change{font-size:.87rem;font-weight:900;margin-top:5px}.up{color:var(--red)}.down{color:var(--green)}.flat{color:var(--muted)}
.signal{display:grid;grid-template-columns:minmax(130px,.34fr) 1fr;gap:12px;background:var(--surface);padding:9px 10px 11px;border-bottom:1px solid var(--line);}
.signal-title{font-size:1.02rem;font-weight:950;color:var(--blue);line-height:1.3}.signal-copy{font-size:.91rem;line-height:1.45;color:#3f434a}.risk{color:#8a3b11}
.quick{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);border-top:0;background:var(--surface);}
.q{padding:9px 10px;border-right:1px solid var(--line);min-width:0}.q:last-child{border-right:0}.qk{font-size:.72rem;color:var(--muted);font-weight:800}.qv{font-size:.94rem;font-weight:900;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.liveflag{display:inline-flex;align-items:center;gap:6px;font-size:.78rem;font-weight:900;color:var(--green);letter-spacing:.04em}.dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 0 3px rgba(7,135,93,.12)}
.offflag{font-size:.78rem;font-weight:850;color:var(--muted)}
.datagrid{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);background:var(--surface);margin:8px 0 12px}.datum{padding:10px;border-right:1px solid var(--line);border-bottom:1px solid var(--line)}.datum:nth-child(4n){border-right:0}.dk{font-size:.74rem;color:var(--muted);font-weight:800}.dv{font-size:1rem;font-weight:950;margin-top:3px}.ds{font-size:.73rem;color:var(--muted);margin-top:2px;line-height:1.3}.flowbar{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border:1px solid var(--line);margin:6px 0 10px}.flowbar>div{background:var(--surface);padding:8px 9px}.flowbar small{display:block;color:var(--muted);font-weight:800}.flowbar b{display:block;margin-top:2px;font-size:.95rem}.overview{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:6px 0 14px}.ovcol{border:1px solid var(--line);background:var(--surface)}.ovhead{padding:10px;border-bottom:1px solid var(--line);font-weight:950}.ovitem{display:grid;grid-template-columns:32px 1fr;gap:8px;padding:10px;border-bottom:1px solid var(--line)}.ovitem:last-child{border-bottom:0}.ovrank{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#9a9b95;font-weight:900}.ovname{font-weight:900}.ovmeta{font-size:.8rem;color:var(--muted);margin-top:2px}.consensus{border:1px solid var(--line);background:var(--surface);padding:10px 11px;margin:10px 0 6px}.consensus b{font-size:1rem}.sectorbar{display:flex;gap:7px;overflow-x:auto;padding:4px 0 10px;margin-bottom:4px;scrollbar-width:none}.sectorbar::-webkit-scrollbar{display:none}.sectorpill{flex:0 0 auto;border:1px solid var(--line);background:var(--surface);padding:7px 10px;border-radius:999px;font-size:.78rem;font-weight:850;color:#3f434a}.sectorpill b{color:var(--ink);margin-left:4px}.sectorpill.hot{border-color:#bfd3f7;background:#f1f6ff;color:#0b57d0}.sectorpill.weak{background:#faf4f0;color:#8a3b11}.consensus-note{font-size:.79rem;color:var(--muted);margin-top:4px}
.plan{display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:var(--line);border:1px solid var(--line);margin-bottom:8px}.plan>div{background:var(--surface);padding:9px}.plan small{display:block;color:var(--muted);font-weight:800}.plan b{display:block;margin-top:3px;font-size:.98rem}
.chartchips{display:flex;gap:12px;align-items:center;font-size:.76rem;color:var(--muted);margin:4px 0 1px;flex-wrap:wrap}.chipline{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:5px}.ma5{background:#111318}.ma20{background:#0b57d0}.ma60{background:#f08c00}.ma120{background:#7b61ff}.ma240{background:#2f9e44}.bb{background:#d9480f}.ind-caption{font-size:.78rem;color:var(--muted);margin:2px 0 6px}
.mini-note{font-size:.78rem;color:var(--muted);line-height:1.4}
[data-testid="stExpander"]{border:0!important;border-bottom:1px solid var(--line)!important;border-radius:0!important;background:transparent!important;}[data-testid="stExpander"] summary{padding-left:0!important;padding-right:0!important;font-weight:850!important;}
div[data-baseweb="tab-list"]{gap:4px}button[data-baseweb="tab"]{font-weight:850}.stRadio [role="radiogroup"]{gap:.35rem!important;align-items:center}.stRadio label{background:var(--surface);border:1px solid var(--line);padding:.16rem .58rem;border-radius:999px}.stRadio label p{color:var(--ink)!important;font-weight:850!important}button[data-baseweb="tab"]{color:var(--ink)!important}
.stButton>button{border-radius:5px!important;font-weight:850!important;min-height:42px}.stButton>button[kind="primary"]{background:var(--ink)!important;border-color:var(--ink)!important;}
hr{border-color:var(--line)!important}
@media(max-width:720px){
 .stApp{font-size:17.5px}.block-container{padding:4.35rem .72rem 3rem}.mast{padding:13px 0 10px;margin-top:.15rem}.brand{font-size:1.52rem}.version{font-size:.68rem}
 .marketbar{font-size:.84rem;gap:8px 11px}.rankrow{grid-template-columns:36px minmax(0,1fr) auto;padding:12px 7px 11px;gap:7px}.rankno{font-size:1.08rem}.stockname{font-size:1.30rem}.stockcode{font-size:.78rem}.industry{font-size:.84rem}.quote{min-width:78px}.price{font-size:1.25rem}.change{font-size:.88rem}
 .signal{grid-template-columns:1fr;gap:3px;padding:9px 7px 10px}.signal-title{font-size:1.05rem}.signal-copy{font-size:.96rem;line-height:1.42}
 .quick{grid-template-columns:repeat(2,1fr)}.q:nth-child(2){border-right:0}.q:nth-child(-n+2){border-bottom:1px solid var(--line)}.qk{font-size:.78rem}.qv{font-size:1rem}
 .datagrid{grid-template-columns:repeat(2,1fr)}.datum:nth-child(odd){border-right:1px solid var(--line)}.datum:nth-child(even){border-right:0}.dk{font-size:.80rem}.dv{font-size:1.05rem}.ds{font-size:.77rem}
 .plan{grid-template-columns:repeat(2,1fr)}.overview{grid-template-columns:1fr}.flowbar{grid-template-columns:repeat(2,1fr)}.section-title{font-size:1.25rem}
 .stRadio [role="radiogroup"]{gap:.22rem!important}.stRadio label{font-size:.92rem!important;padding:.14rem .45rem!important}
 @supports (padding-top:env(safe-area-inset-top)){.block-container{padding-top:calc(4.35rem + env(safe-area-inset-top));}}
}
</style>
"""


def esc(v) -> str:
    return html.escape(str(v if v is not None else ""))


def finite(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def nfmt(v, d=2):
    x = finite(v)
    if x is None:
        return "—"
    return f"{x:,.{d}f}".rstrip("0").rstrip(".")


def pp(v, d=1):
    x = finite(v)
    return "—" if x is None else f"{x:+.{d}f}%"


def decpct(v, d=1):
    x = finite(v)
    return "—" if x is None else f"{x*100:+.{d}f}%"


def lots(v):
    x = finite(v)
    return "—" if x is None else f"{x:+,.0f}張"


def _research(stock: dict):
    r = stock.get("research") or {}
    return (
        r.get("monthly_revenue") or {},
        r.get("financials") or {},
        r.get("valuation") or {},
        r.get("institutional_flow") or {},
        r.get("main_force_proxy") or {},
    )


def _quick_values(stock: dict):
    rev, fin, val, inst, _ = _research(stock)
    latest = rev.get("latest") or {}
    eps4 = fin.get("eps_4q_sum")
    if eps4 is not None:
        eps_label, eps_value, eps_note = "近4季EPS", nfmt(eps4, 2), "單季加總"
    elif fin.get("latest_ytd_eps") is not None:
        eps_label, eps_value, eps_note = "最新累計EPS", nfmt(fin.get("latest_ytd_eps"), 2), fin.get("latest_period") or "官方季報"
    else:
        eps_label, eps_value, eps_note = "EPS", "—", "資料未取得"
    sector = stock.get("sector_strength") or {}
    theme = stock.get("theme_strength") or {}
    mainline = finite(((stock.get("horizons") or {}).get("mid",{}).get("score_components") or {}).get("mainline"))
    if mainline is None:
        mainline = 0.56*(finite(sector.get("score")) or 50.0)+0.44*(finite(theme.get("score")) or 50.0)
    theme_name = str(theme.get("name") or stock.get("theme") or sector.get("status") or "主線")
    return [
        ("月營收", f"{nfmt(latest.get('revenue_billion'),1)}億", f"YoY {pp(latest.get('yoy_pct'))}"),
        (eps_label, eps_value, eps_note),
        ("產業鏈主線", nfmt(mainline, 0), theme_name),
        ("近月法人", lots(inst.get("total_net_lots")) if inst.get("available") else "—", "合計"),
    ]


def _quick_html(stock: dict) -> str:
    cells = []
    for k, v, s in _quick_values(stock):
        cells.append(f"<div class='q'><div class='qk'>{esc(k)}</div><div class='qv'>{esc(v)}</div><div class='ds'>{esc(s)}</div></div>")
    return "<div class='quick'>" + "".join(cells) + "</div>"


def _render_fundamental_charts(stock: dict, key: str):
    rev, fin, _, inst, _ = _research(stock)
    rows = rev.get("rows") or []
    if rows:
        rdf = pd.DataFrame(rows[-24:])
        if not rdf.empty and {"month","revenue_billion"}.issubset(rdf.columns):
            fig = go.Figure()
            fig.add_bar(x=rdf["month"], y=pd.to_numeric(rdf["revenue_billion"], errors="coerce"), name="月營收")
            fig.update_layout(height=250, margin=dict(l=5,r=5,t=8,b=5), showlegend=False, paper_bgcolor="#fffefb", plot_bgcolor="#fffefb", xaxis_title="", yaxis_title="億元")
            fig.update_xaxes(nticks=6, fixedrange=True)
            fig.update_yaxes(gridcolor="#ecebe5", fixedrange=True)
            st.plotly_chart(fig, use_container_width=True, key=f"{key}_rev", config={"displayModeBar":False})
    qhist = fin.get("history_quarters") or fin.get("quarters") or []
    if qhist:
        qdf = pd.DataFrame(qhist[-8:])
        if not qdf.empty and {"quarter","eps"}.issubset(qdf.columns):
            fig = go.Figure()
            fig.add_bar(x=qdf["quarter"], y=pd.to_numeric(qdf["eps"], errors="coerce"), name="EPS")
            fig.update_layout(height=240, margin=dict(l=5,r=5,t=8,b=5), showlegend=False, paper_bgcolor="#fffefb", plot_bgcolor="#fffefb", xaxis_title="", yaxis_title="EPS")
            fig.update_xaxes(fixedrange=True)
            fig.update_yaxes(gridcolor="#ecebe5", fixedrange=True)
            st.plotly_chart(fig, use_container_width=True, key=f"{key}_eps", config={"displayModeBar":False})
    daily = inst.get("daily_rows") or []
    if daily:
        ddf = pd.DataFrame(daily[-15:])
        st.markdown("#### 近月法人日別")
        st.dataframe(ddf.rename(columns={"date":"日期","foreign_net_lots":"外資(張)","trust_net_lots":"投信(張)","dealer_net_lots":"自營商(張)","total_net_lots":"合計(張)"}), hide_index=True, use_container_width=True)


def _render_research(stock: dict):
    rev, fin, val, inst, main = _research(stock)
    latest = rev.get("latest") or {}
    qs = fin.get("quarters") or []
    gm = fin.get("gross_margin_latest_pct")
    eps_name = "近4季EPS" if fin.get("eps_4q_sum") is not None else "最新累計EPS" if fin.get("latest_ytd_eps") is not None else "EPS"
    eps_value = fin.get("eps_4q_sum") if fin.get("eps_4q_sum") is not None else fin.get("latest_ytd_eps")
    eps_note = "單季加總" if fin.get("eps_4q_sum") is not None else (fin.get("latest_period") or "官方季報") if fin.get("latest_ytd_eps") is not None else "資料未取得"
    items = [
        ("最新月營收", f"{nfmt(latest.get('revenue_billion'))} 億", latest.get("month") or "—"),
        ("月增", pp(latest.get("mom_pct")), "MoM"), ("年增", pp(latest.get("yoy_pct")), "YoY"),
        (eps_name, nfmt(eps_value), eps_note), ("最新毛利率", pp(gm, 1) if gm is not None else "—", ""),
        ("本益比", f"{nfmt(val.get('pe'),1)} 倍" if val.get("pe") is not None else "—", val.get("date") or ""),
        ("股價淨值比", nfmt(val.get("pbr"),1), "PBR"), ("殖利率", pp(val.get("dividend_yield_pct"),1), "近值"),
        ("近4季平均毛利", pp(fin.get("gross_margin_4q_avg_pct"),1) if fin.get("gross_margin_4q_avg_pct") is not None else "—", ""),
        ("資料時間", stock.get("price_date") or "—", "股價基準"),
    ]
    html_cells="".join(f"<div class='datum'><div class='dk'>{esc(k)}</div><div class='dv'>{esc(v)}</div><div class='ds'>{esc(note)}</div></div>" for k,v,note in items)
    st.markdown(f"<div class='datagrid'>{html_cells}</div>", unsafe_allow_html=True)
    st.markdown(f"""<div class='flowbar'>
      <div><small>外資</small><b>{esc(lots(inst.get('foreign_net_lots')) if inst.get('available') else '—')}</b></div>
      <div><small>投信</small><b>{esc(lots(inst.get('trust_net_lots')) if inst.get('available') else '—')}</b></div>
      <div><small>自營商</small><b>{esc(lots(inst.get('dealer_net_lots')) if inst.get('available') else '—')}</b></div>
      <div><small>三大法人合計</small><b>{esc(lots(inst.get('total_net_lots')) if inst.get('available') else '—')}</b></div>
    </div>""", unsafe_allow_html=True)
    if qs:
        st.dataframe(pd.DataFrame([{"季度":q.get("quarter"),"EPS":q.get("eps"),"同年度累積EPS*":q.get("eps_ytd_sum_unadjusted"),"毛利率(%)":q.get("gross_margin_pct")} for q in qs[-4:]]),hide_index=True,use_container_width=True)
        st.caption("* 依單季 EPS 加總；遇配股/分割可能與公司重編累計值不同。")
    source=str((stock.get("research") or {}).get("source") or "")
    if source: st.caption("公司資料來源："+source)
    if rev.get("history_limited") or fin.get("history_limited"):
        st.caption("目前使用官方最新期備援資料；歷史不足時模型會降低基本面信心。")
    if inst.get("summary"): st.caption(str(inst.get("summary")))
    key=str(stock.get("ticker") or "x").replace(".","_")
    if st.toggle("顯示營收 / EPS / 法人明細", value=False, key=f"fund_toggle_{key}"):
        _render_fundamental_charts(stock,key)


def _chart_dataframe(chart: dict) -> pd.DataFrame:
    if not chart or "ohlcv" not in chart:
        return pd.DataFrame()
    df = pd.DataFrame(chart["ohlcv"], columns=["Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(chart.get("dates", []), errors="coerce")
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    if df.empty:
        return df
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    c = df["Close"]
    for n in [5, 20, 60, 120, 240]:
        df[f"MA{n}"] = c.rolling(n).mean()
    m20 = c.rolling(20).mean()
    s20 = c.rolling(20).std()
    df["BBU"] = m20 + 2 * s20
    df["BBL"] = m20 - 2 * s20

    low9 = df["Low"].rolling(9).min()
    high9 = df["High"].rolling(9).max()
    k = (c - low9) / (high9 - low9).replace(0, np.nan) * 100
    df["KD_K"] = k.ewm(alpha=1/3, adjust=False).mean()
    df["KD_D"] = df["KD_K"].ewm(alpha=1/3, adjust=False).mean()

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]
    return df


def render_chart(chart: dict, plan: dict | None, key: str):
    raw = _chart_dataframe(chart)
    if raw.empty:
        st.caption("K線資料未取得。")
        return
    window = st.radio("區間", ["1個月", "3個月", "6個月", "1年"], index=2, horizontal=True, label_visibility="collapsed", key=f"{key}_range")
    n = {"1個月": 24, "3個月": 66, "6個月": 132, "1年": 252}[window]
    base = raw.tail(n).copy()
    opts = list(base["Date"])
    if len(opts) >= 2:
        left, right = st.select_slider(
            "放大區間", options=opts, value=(opts[0], opts[-1]),
            format_func=lambda d: pd.Timestamp(d).strftime("%m/%d"),
            key=f"{key}_zoom_{window}", help="拖曳左右端點可鎖定想看的交易日期；不使用手勢平移，避免手機上整張圖跑掉。"
        )
        df = base[(base["Date"] >= pd.Timestamp(left)) & (base["Date"] <= pd.Timestamp(right))].copy()
    else:
        df = base.copy()
    x = df["Date"].dt.strftime("%Y-%m-%d").tolist()
    if df.empty:
        st.caption("所選區間沒有交易資料。")
        return

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.54,0.16,0.15,0.15], vertical_spacing=0.025)
    fig.add_trace(go.Candlestick(
        x=x, open=df.Open, high=df.High, low=df.Low, close=df.Close,
        increasing_line_color="#d92d20", increasing_fillcolor="#d92d20",
        decreasing_line_color="#07875d", decreasing_fillcolor="#07875d",
        name="價格", showlegend=False, whiskerwidth=.72,
    ), row=1,col=1)
    fig.add_trace(go.Scatter(x=x,y=df.BBU,mode="lines",line=dict(color="rgba(217,72,15,.55)",width=1.1,dash="dot"),hovertemplate="布林上緣 %{y:.2f}<extra></extra>"),row=1,col=1)
    fig.add_trace(go.Scatter(x=x,y=df.BBL,mode="lines",line=dict(color="rgba(217,72,15,.55)",width=1.1,dash="dot"),fill='tonexty',fillcolor='rgba(217,72,15,.045)',hovertemplate="布林下緣 %{y:.2f}<extra></extra>"),row=1,col=1)
    ma_specs=[("MA5","#111318",1.35),("MA20","#0b57d0",1.8),("MA60","#f08c00",1.65),("MA120","#7b61ff",1.35),("MA240","#2f9e44",1.35)]
    for col,color,width in ma_specs:
        fig.add_trace(go.Scatter(x=x,y=df[col],mode="lines",line=dict(color=color,width=width),hovertemplate=f"{col.replace('MA','')}日 %{{y:.2f}}<extra></extra>"),row=1,col=1)
    colors=np.where(df.Close>=df.Open,"rgba(217,45,32,.62)","rgba(7,135,93,.62)")
    fig.add_trace(go.Bar(x=x,y=df.Volume,marker_color=colors,showlegend=False,hovertemplate="量 %{y:,.0f}<extra></extra>"),row=2,col=1)
    fig.add_trace(go.Scatter(x=x,y=df.KD_K,mode='lines',line=dict(color='#0b57d0',width=1.6),hovertemplate='K %{y:.1f}<extra></extra>'),row=3,col=1)
    fig.add_trace(go.Scatter(x=x,y=df.KD_D,mode='lines',line=dict(color='#f08c00',width=1.6),hovertemplate='D %{y:.1f}<extra></extra>'),row=3,col=1)
    fig.add_hline(y=80,line_color='rgba(120,120,120,.40)',line_width=1,line_dash='dot',row=3,col=1); fig.add_hline(y=20,line_color='rgba(120,120,120,.40)',line_width=1,line_dash='dot',row=3,col=1)
    macd_colors=np.where(df.MACD_HIST>=0,'rgba(217,45,32,.55)','rgba(7,135,93,.55)')
    fig.add_trace(go.Bar(x=x,y=df.MACD_HIST,marker_color=macd_colors,showlegend=False,hovertemplate='MACD柱 %{y:.3f}<extra></extra>'),row=4,col=1)
    fig.add_trace(go.Scatter(x=x,y=df.MACD,mode='lines',line=dict(color='#111318',width=1.4),hovertemplate='MACD %{y:.3f}<extra></extra>'),row=4,col=1)
    fig.add_trace(go.Scatter(x=x,y=df.MACD_SIGNAL,mode='lines',line=dict(color='#7b61ff',width=1.4),hovertemplate='訊號 %{y:.3f}<extra></extra>'),row=4,col=1)
    if plan:
        stop=finite(plan.get("invalidation")); trigger=finite(plan.get("trigger"))
        if trigger is not None: fig.add_hline(y=trigger,line_color="#8a5a00",line_width=1.1,line_dash="dot",row=1,col=1)
        if stop is not None: fig.add_hline(y=stop,line_color="#9b1c1c",line_width=1.1,line_dash="dash",row=1,col=1)
    vals=pd.concat([df["High"],df["Low"],df["MA20"],df["MA60"],df["MA120"],df["MA240"],df["BBU"],df["BBL"]]).dropna()
    if not vals.empty:
        lo,hi=float(vals.min()),float(vals.max()); pad=max((hi-lo)*.14,max(hi,1)*.010)
        fig.update_yaxes(range=[max(0,lo-pad),hi+pad],row=1,col=1)
    # Category x-axis removes weekend/holiday gaps, keeping candles almost touching.
    if x:
        ids=sorted(set(np.linspace(0,len(x)-1,min(4,len(x)),dtype=int).tolist()))
        tickvals=[x[i] for i in ids]; ticktext=[pd.to_datetime(x[i]).strftime("%m/%d") for i in ids]
        for r in [1,2,3,4]:
            fig.update_xaxes(type="category",tickmode="array",tickvals=tickvals,ticktext=ticktext,fixedrange=True,row=r,col=1)
    fig.update_layout(height=575,margin=dict(l=3,r=5,t=6,b=2),paper_bgcolor="#fffefb",plot_bgcolor="#fffefb",xaxis_rangeslider_visible=False,dragmode=False,hovermode="x unified",showlegend=False,font=dict(size=11,color="#4d5159"),bargap=.04)
    fig.update_yaxes(gridcolor="#ecebe5",zeroline=False,fixedrange=True,tickfont=dict(size=10))
    st.markdown("<div class='chartchips'><span>K線（日）</span><span><i class='chipline ma5'></i>週線5</span><span><i class='chipline ma20'></i>月線20</span><span><i class='chipline ma60'></i>季線60</span><span><i class='chipline ma120'></i>半年120</span><span><i class='chipline ma240'></i>年線240</span><span><i class='chipline bb'></i>布林</span></div><div class='ind-caption'>下方依序：量能、KD、MACD；交易日採等距顯示，週末不留大空隙。</div>",unsafe_allow_html=True)
    st.plotly_chart(fig,use_container_width=True,key=key,config={"displayModeBar":False,"scrollZoom":False,"displaylogo":False,"staticPlot":False})


def _history_line(stock: dict, h: str) -> str:
    f = stock.get("horizons", {}).get(h, {}).get("forecast", {}) or {}
    s = f.get("strategy") or {}
    if not f.get("estimate_available"):
        return "歷史案例不足"
    med = decpct(s.get("median"), 1)
    rate = finite(s.get("smoothed_positive_rate"))
    n = int(finite(f.get("local_effective_n")) or 0)
    return f"相似案例 {n} 次 · 中位數 {med} · 偏正向 {rate*100:.0f}%" if rate is not None else f"相似案例 {n} 次 · 中位數 {med}"


def _row(stock: dict, h: str, snap: dict, rank: int, live: bool = False):
    name = stock.get("name") or stock.get("ticker")
    ticker = stock.get("ticker") or ""
    fine = stock.get("fine_industry") or fine_industry(ticker, name, stock.get("industry", ""))
    role_label = ((stock.get("role") or {}).get("label") or "").strip()
    stage_label = ((stock.get("stage") or {}).get("label") or "").strip()
    tags = [f"<span class='tag'>{esc(fine)}</span>"] if fine else []
    if role_label:
        tags.append(f"<span class='tag blue'>{esc(role_label)}</span>")
    if stage_label:
        tags.append(f"<span class='tag red'>{esc(stage_label)}</span>")
    tag_html = "<div class='tagrow'>" + "".join(tags) + "</div>" if tags else ""
    if live:
        live_info = stock.get("intraday") or {}
        view = intraday_view(stock)
        price = live_info.get("price") or stock.get("price")
        change = finite(live_info.get("change_rate_pct"))
        reasons = view.get("reasons") or []
        title = view.get("title") or "即時資料不足"
        risk = ""
        history = _history_line(stock, h)
    else:
        view = investment_view(stock, h, snap.get("market") or {})
        price = stock.get("price")
        change = None
        reasons = view.get("reasons") or []
        title = view.get("title") or "觀察"
        risk = view.get("risk") or ""
        history = _history_line(stock, h)

    cls = "flat"
    if change is not None:
        cls = "up" if change > 0 else "down" if change < 0 else "flat"
    change_text = pp(change, 2) if change is not None else esc(H_LABEL[h])
    reason_text = " · ".join(str(x) for x in reasons[:2])
    risk_text = f" · <span class='risk'>{esc(risk)}</span>" if risk else ""
    st.markdown(f"""
    <div class="rankrow">
      <div class="rankno">{rank:02d}</div>
      <div><div><span class="stockname">{esc(name)}</span><span class="stockcode">{esc(ticker)}</span></div>{tag_html}</div>
      <div class="quote"><div class="price">{nfmt(price)}</div><div class="change {cls}">{esc(change_text)}</div></div>
    </div>
    <div class="signal"><div class="signal-title">{esc(title)}</div><div class="signal-copy">{esc(reason_text)}{risk_text}<div class="mini-note">{esc(history)}</div></div></div>
    {_quick_html(stock)}
    """, unsafe_allow_html=True)

    label = f"{name}｜數據與走勢"
    with st.expander(label, expanded=False):
        _render_research(stock)
        block = stock.get("horizons", {}).get(h, {}) or {}
        plan = block.get("plan") or {}
        if plan:
            pos = block.get("position_guidance") or {}
            pos_label = pos.get("label") or "依風險調整"
            st.markdown(f"""<div class='plan'>
              <div><small>較理想區間</small><b>{nfmt(plan.get('zone_low'))}–{nfmt(plan.get('zone_high'))}</b></div>
              <div><small>轉強確認</small><b>{nfmt(plan.get('trigger'))}</b></div>
              <div><small>不追高超過</small><b>{nfmt(plan.get('chase_limit'))}</b></div>
              <div><small>跌破重看</small><b>{nfmt(plan.get('invalidation'))}</b></div>
              <div><small>目前部位節奏</small><b>{esc(pos_label)}</b></div>
            </div>""", unsafe_allow_html=True)
        if st.toggle("顯示 K 線與技術指標", value=False, key=f"chart_toggle_{'live' if live else h}_{ticker}_{snap.get('snapshot_id','x')}"):
            chart = None
            try:
                chart = service.chart_on_demand(snap, ticker, DATA_DIR, allow_fetch=False)
            except Exception:
                pass
            render_chart(chart, plan, f"chart_{'live' if live else h}_{ticker}_{snap.get('snapshot_id','x')}")


def _render_daily_list(snap: dict | None, h: str, n: int = 5):
    if not snap:
        st.info("請先更新市場資料。")
        return
    picks = service.select_market_best(snap, h, n=n)
    if not picks:
        st.info("目前沒有可排序標的。")
        return
    for i, stock in enumerate(picks, 1):
        _row(stock, h, snap, i, live=False)


def _market_header(snap: dict | None):
    if not snap:
        st.markdown("<div class='marketbar'><span>尚未建立市場快照</span></div>", unsafe_allow_html=True)
        return
    m=snap.get("market") or {}; cov=snap.get("coverage") or {}; off=cov.get("official_eod") or {}
    generated=str(snap.get("generated_at") or st.session_state.get("alpha_last_refresh") or "")
    if "T" in generated: generated=generated.replace("T"," ")[:19]
    parts=[f"<span><b>基準 {esc(snap.get('price_date') or '—')}</b> 完整日線</span>",f"<span>{esc(REGIME.get(m.get('regime'),'大盤待確認'))}</span>",f"<span>上市 {esc(off.get('twse_date') or '—')}</span>",f"<span>上櫃 {esc(off.get('tpex_date') or '—')}</span>"]
    if generated: parts.append(f"<span>模型更新 {esc(generated)}</span>")
    if market_is_open(_taipei_timestamp()): parts.append("<span class='liveflag'><i class='dot'></i>盤中即時自動重排</span>")
    stale=int(cov.get("stale_excluded") or 0)
    if stale: parts.append(f"<span>{stale:,} 檔舊資料已排除</span>")
    rc=cov.get("recommendation_research") or {}
    if rc.get("selected"):
        parts.append(f"<span>營收 {int(rc.get('revenue') or 0)}/{int(rc.get('selected') or 0)}</span>")
        parts.append(f"<span>財報 {int(rc.get('financials') or 0)}/{int(rc.get('selected') or 0)}</span>")
    errors=cov.get("errors") or []
    if errors: parts.append(f"<span>資料警示 {len(errors)}</span>")
    st.markdown("<div class='marketbar'>"+"".join(parts)+"</div>",unsafe_allow_html=True)
    if market_is_open(_taipei_timestamp()):
        st.caption("盤中模式不把未收盤的半根日K塞進長期模型；以最近完整交易日為基準，再用即時價量重新排序。")
    if errors:
        with st.expander("資料來源狀態",expanded=False):
            for err in errors[:8]: st.caption(str(err))


def _intraday_render_once(snap: dict, horizon: str = "short", top_n: int = 8):
    now=_taipei_timestamp()
    if horizon not in {"short","mid","long"}: horizon="short"
    if not market_is_open(now):
        label="盤前" if now.hour<9 else "收盤後"
        st.markdown(f"<div class='offflag'>{label} · {esc(now.strftime('%Y-%m-%d %H:%M:%S'))} · 09:00–13:30 才啟用盤中排序</div>",unsafe_allow_html=True)
        _render_daily_list(snap,horizon,n=min(top_n,5)); return
    client=FinMindRealtimeClient()
    tickers=candidate_tickers(snap,per_horizon=40)
    df,status=client.snapshots(tickers)
    if not status.available:
        st.warning(f"盤中即時資料暫不可用：{status.reason}；目前顯示最近完整日線排名。")
        _render_daily_list(snap,horizon,n=min(top_n,5)); return
    live=rerank_snapshot(snap,horizon,df,top_n=top_n)
    twii=df[df["stock_id"].astype(str)=="001"] if "stock_id" in df.columns else pd.DataFrame()
    idx_change=finite(twii.iloc[-1].get("change_rate")) if not twii.empty else None
    fetched=(status.fetched_at or now.isoformat(timespec="seconds")).replace("T"," ")[:19]
    st.markdown(
        f"<div class='marketbar'><span class='liveflag'><i class='dot'></i>盤中即時</span>"
        f"<span>{esc(fetched)}</span><span>來源 {esc(status.source)}</span>"
        f"<span>基準日線 {esc(snap.get('price_date') or '—')}</span>"
        f"<span>加權 {esc(pp(idx_change,2))}</span><span>{esc(H_LABEL[horizon])}</span></div>",
        unsafe_allow_html=True,
    )
    if not live: st.info("即時資料已取得，但候選股目前無可排序資料。"); return
    for i,stock in enumerate(live,1): _row(stock,horizon,snap,i,live=True)


# Only this fragment refreshes; the expensive multi-year model remains untouched.
if hasattr(st,"fragment"):
    @st.fragment(run_every=60)
    def intraday_fragment(snap: dict, horizon: str = "short"):
        _intraday_render_once(snap,horizon)
else:
    def intraday_fragment(snap: dict, horizon: str = "short"):
        _intraday_render_once(snap,horizon)



def _live_rankings(snap: dict | None, top_n: int = 5):
    """Fetch one realtime batch and return live rankings for all horizons."""
    if not snap or not market_is_open(_taipei_timestamp()):
        return None, None
    client=FinMindRealtimeClient()
    tickers=candidate_tickers(snap,per_horizon=45)
    df,status=client.snapshots(tickers)
    if not status.available or df.empty:
        return None, status
    out={h: rerank_snapshot(snap,h,df,top_n=top_n) for h in ["short","mid","long"]}
    if not any(out.values()):
        return None, status
    return out, status


def _overview(snap: dict | None, allow_live: bool = True):
    if not snap:
        st.info("請先更新市場資料。")
        return

    # Two-layer market mainline: official sectors + cross-industry supply chains.
    themes = snap.get("theme_leadership") or {}
    sectors = snap.get("sector_leadership") or {}
    pills=[]
    source_map = themes if themes else sectors
    for industry, info in list(source_map.items())[:7]:
        score=finite((info or {}).get("score")) or 50.0
        cls="hot" if score>=68 else "weak" if score<40 else ""
        pills.append(f"<span class='sectorpill {cls}'>{esc(industry)} <b>{score:.0f}</b></span>")
    if pills:
        st.markdown("<div class='section-kicker'>MARKET MAINLINE</div><div class='sectorbar'>"+''.join(pills)+"</div>",unsafe_allow_html=True)

    mapping = [("short", "短線 · 約2週"), ("mid", "中線 · 約2個月"), ("long", "長線 · 約6個月")]
    raw = {h: service.select_market_best(snap, h, n=5) for h, _ in mapping}
    live_status=None
    if allow_live and market_is_open(_taipei_timestamp()):
        live_raw, live_status=_live_rankings(snap,top_n=5)
        if live_raw:
            raw=live_raw
            fetched=(live_status.fetched_at or _taipei_timestamp().isoformat(timespec="seconds")).replace("T"," ")[:19]
            st.markdown(
                f"<div class='marketbar'><span class='liveflag'><i class='dot'></i>盤中總覽已即時重排</span>"
                f"<span>{esc(fetched)}</span><span>來源 {esc(live_status.source)}</span>"
                f"<span>基準完整日線 {esc(snap.get('price_date') or '—')}</span></div>",
                unsafe_allow_html=True,
            )
        elif live_status is not None:
            st.warning(f"盤中行情暫不可用：{live_status.reason}；以下暫顯示 {snap.get('price_date') or '最近完整交易日'} 的基準排名。")

    cols_html = []
    dup_counts = {}
    for h, title in mapping:
        picks = raw.get(h, [])[:5]
        body = []
        for i, stock in enumerate(picks, 1):
            code = str(stock.get('ticker') or '')
            dup_counts[code] = dup_counts.get(code, 0) + 1
            name = str(stock.get('name') or code)
            fine = str(stock.get('fine_industry') or stock.get('industry') or '')
            live_score=finite(stock.get('live_ranking_score'))
            score = live_score if live_score is not None else finite((stock.get('horizons', {}).get(h, {}) or {}).get('ranking_score'))
            change=finite((stock.get('intraday') or {}).get('change_rate_pct'))
            comp=((stock.get('horizons',{}).get(h,{}) or {}).get('score_components') or {})
            mainline_score=finite(comp.get('mainline'))
            meta=f"{fine} · {'盤中' if live_score is not None else '基準'}分數 {nfmt(score,1)}"
            if change is not None:
                meta += f" · 今日 {change:+.2f}%"
            elif mainline_score is not None:
                meta += f" · 主線 {mainline_score:.0f}"
            body.append(f"<div class='ovitem'><div class='ovrank'>#{i}</div><div><div class='ovname'>{esc(name)} <span class='stockcode'>{esc(code)}</span></div><div class='ovmeta'>{esc(meta)}</div></div></div>")
        cols_html.append(f"<div class='ovcol'><div class='ovhead'>{esc(title)}</div>{''.join(body)}</div>")

    consensus = []
    for code, cnt in sorted(dup_counts.items(), key=lambda x: (-x[1], x[0])):
        if cnt >= 2:
            stock = next((s for s in snap.get('stocks', []) if s.get('ticker') == code), None)
            if stock:
                consensus.append(f"{stock.get('name')}（{cnt}/3）")
    if consensus:
        st.markdown("<div class='consensus'><b>跨週期共識</b><div class='mini-note'>" + esc('、'.join(consensus[:6])) + "</div><div class='consensus-note'>共識代表同一檔在不同時間尺度都強；實際配置時仍只算一個部位。</div></div>", unsafe_allow_html=True)

    unique_n = len(set(code for code in dup_counts))
    mode_text="盤中即時訊號" if any((s.get('intraday') for picks in raw.values() for s in picks)) else "完整日線訊號"
    st.markdown(f"<div class='mini-note'>{mode_text}｜15 個訊號席位共 {unique_n} 檔股票。</div>", unsafe_allow_html=True)
    st.markdown("<div class='overview'>" + ''.join(cols_html) + "</div>", unsafe_allow_html=True)

    # Portfolio diversification is separate from signal ranking.  Keep using the
    # daily allocation helper so a few minutes of price action do not rewrite the
    # strategic portfolio constraints.
    allocation = service.select_cross_horizon_shortlists(snap, n=5, max_appearances=2)
    metrics = service.shortlist_overlap_metrics(allocation)
    st.markdown(f"<div class='mini-note'>若用於實戰配置：去重後約 {metrics.get('unique_tickers',0)} 檔不同股票，單一股票最多跨 2 個週期。</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-kicker'>PRIME</div><div class='section-title'>各週期首選</div>", unsafe_allow_html=True)
    _prime(snap, raw)


def _prime(snap: dict | None, joint: dict | None = None):
    if not snap:
        st.info("請先更新市場資料。")
        return
    joint = joint or {h: service.select_market_best(snap, h, n=5) for h in HORIZONS}
    used: set[str] = set()
    for h in HORIZONS:
        picks = joint.get(h, [])
        stock = next((x for x in picks if x.get("ticker") not in used), picks[0] if picks else None)
        if stock:
            used.add(str(stock.get("ticker")))
            st.markdown(f"<div class='section-kicker'>{esc(H_LABEL[h])}</div>", unsafe_allow_html=True)
            _row(stock, h, snap, 1, live=bool(stock.get("intraday")))


if hasattr(st,"fragment"):
    @st.fragment(run_every=60)
    def overview_fragment(snap: dict):
        _overview(snap,allow_live=True)
else:
    def overview_fragment(snap: dict):
        _overview(snap,allow_live=True)


def _doctor(snap: dict | None):
    with st.form("doctor_v133"):
        code = st.text_input("股票代碼", value="2330")
        h = st.selectbox("時間", list(H_LABEL), format_func=lambda x: H_LABEL[x], index=1)
        own = st.checkbox("已有持股")
        c1, c2 = st.columns(2)
        with c1:
            cost = st.number_input("成本", min_value=0.0, value=0.0)
            stop = st.number_input("原防守價", min_value=0.0, value=0.0)
        with c2:
            trail = st.number_input("獲利保護價", min_value=0.0, value=0.0)
            thesis = st.selectbox("原買進理由", ["尚未確認", "仍成立", "已不成立"])
        go_ = st.form_submit_button("查看", type="primary", use_container_width=True)
    if go_:
        dr = service.diagnose(code, snap, DATA_DIR)
        dr["h"] = h
        st.session_state["alpha_doctor"] = dr
    dr = st.session_state.get("alpha_doctor")
    if not dr:
        return
    if dr.get("error"):
        st.error("找不到足夠行情資料。")
        return
    stock = dr.get("stock")
    if not stock:
        return
    _row(stock, dr.get("h", "mid"), snap or {"market": {}, "snapshot_id": "doctor"}, 1, live=False)
    if own:
        result = holding_review(
            stock.get("price", 0),
            original_invalidation=stop if stop > 0 else None,
            trailing_protection=trail if trail > 0 else None,
            thesis_broken=(thesis == "已不成立"),
        )
        pnl = (float(stock.get("price")) / cost - 1) * 100 if cost > 0 and finite(stock.get("price")) is not None else None
        st.info(HOLD_LABELS.get(result, result) + (f" 目前損益約 {pnl:+.1f}%" if pnl is not None else ""))


def main():
    st.set_page_config(page_title="Alpha Radar 16.1", page_icon="◼", layout="wide", initial_sidebar_state="collapsed")
    _bootstrap_secrets()
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("<div class='mast'><div class='brand'>ALPHA<span>/TW</span></div><div class='version'>LIVE 16.1</div></div>", unsafe_allow_html=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if st.session_state.get("alpha_version") != service.OPERATIONS_VERSION:
        for k in ["alpha_snapshot", "alpha_doctor", "alpha_error"]:
            st.session_state.pop(k, None)
        st.session_state["alpha_version"] = service.OPERATIONS_VERSION
        previous = service.load_dashboard(DATA_DIR / "dashboard_snapshot.json")
        if previous and previous.get("model_version") == service.OPERATIONS_VERSION:
            st.session_state["alpha_snapshot"] = service.compact_session_dashboard(previous)

    with st.sidebar:
        st.markdown("### 設定")
        family_label = st.selectbox("資料範圍", list(FAMILY_LABELS), index=0)
        period = st.selectbox("歷史長度", ["5y", "8y", "10y", "3y"], index=0)
        candidate_size = st.selectbox("比較股票數", [400, 700, 1000], index=2)
        research_pool = st.selectbox("公司資料檔數", [5, 10, 15], index=1)
        status = provider_runtime_status()
        st.caption("盤中：自動抓取 TWSE/TPEx MIS 即時行情；若有 FinMind 即時權限則優先使用。")
        st.caption("總覽與各週期頁盤中約每 60 秒自動重排，不需重跑多年模型。")
        if st.button("清除快取"):
            DailyPriceStore(DATA_DIR / "daily_prices.sqlite").clear()
            service.remove_saved_dashboard(DATA_DIR / "dashboard_snapshot.json")
            st.session_state.pop("alpha_snapshot", None)
            st.success("已清除")

    settings = service.RunSettings(
        reference_size=600,
        candidate_size=int(candidate_size),
        research_pool_per_horizon=int(research_pool),
        history_period=period,
        model_family=FAMILY_LABELS[family_label],
    )

    update_label = "更新基準模型（盤中保留完整昨收）" if market_is_open(_taipei_timestamp()) else "更新最新完整日線與模型"
    if st.button(update_label, type="primary", use_container_width=True):
        started = time.monotonic()
        bar = st.progress(0, text="0% · 準備更新")
        try:
            with scan_lock(DATA_DIR / "market_scan.lock"):
                def update(stage, value):
                    v = min(1.0, max(0.0, float(value)))
                    elapsed = int(time.monotonic() - started)
                    heartbeat = " · 仍在處理" if 0.18 <= v < 0.98 else ""
                    bar.progress(v, text=f"{int(v*100):d}% · {stage} · {elapsed}s{heartbeat}")
                snap = service.run_scan(DATA_DIR, settings, progress=update)
                st.session_state["alpha_snapshot"] = service.compact_session_dashboard(snap)
                st.session_state.pop("alpha_error", None)
                st.session_state["alpha_last_refresh"] = _taipei_timestamp().strftime("%Y-%m-%d %H:%M:%S")
                gc.collect()
                bar.progress(1.0, text=f"100% · 更新完成 · {int(time.monotonic()-started)}s")
                st.success("更新完成；基準模型仍使用最近完整交易日，盤中總覽會自動套用即時行情重新排序。" if market_is_open(_taipei_timestamp()) else "更新完成；已保留最新成功快照。")
        except ScanBusyError as exc:
            st.warning(str(exc) + "。本頁會繼續使用上一份成功快照。")
        except Exception as exc:
            token = os.getenv("FINMIND_TOKEN", "")
            st.session_state["alpha_error"] = safe_error_text(exc, mask_tokens=(token,))
            st.error("更新未完成，已保留上一份成功快照。")
        finally:
            time.sleep(0.15)
            bar.empty()

    snap = st.session_state.get("alpha_snapshot")
    _market_header(snap)
    if st.session_state.get("alpha_error"):
        st.error(st.session_state.get("alpha_error"))

    view = st.radio("功能", VIEWS, horizontal=True, label_visibility="collapsed", key="view_v16")
    if view == "總覽":
        st.markdown("<div class='section-kicker'>OVERVIEW</div><div class='section-title'>全景總覽</div>", unsafe_allow_html=True)
        if snap and market_is_open(_taipei_timestamp()):
            overview_fragment(snap)
        else:
            _overview(snap,allow_live=False)
    elif view == "即時":
        st.markdown("<div class='section-kicker'>LIVE PULSE</div><div class='section-title'>盤中雷達</div>", unsafe_allow_html=True)
        if not snap:
            st.info("先建立基準模型，盤中才有候選池。")
        else:
            live_h = st.radio("即時週期", ["short","mid","long"], horizontal=True, format_func=lambda h:H_LABEL[h], label_visibility="collapsed", key="live_h_v16")
            intraday_fragment(snap, live_h)
    elif view in {"短線", "中線", "長線"}:
        h = {"短線": "short", "中線": "mid", "長線": "long"}[view]
        st.markdown(f"<div class='section-kicker'>{esc(H_LABEL[h])}</div><div class='section-title'>前 5 名</div>", unsafe_allow_html=True)
        if snap and market_is_open(_taipei_timestamp()):
            use_live=st.toggle("盤中即時覆蓋", value=True, key=f"live_overlay_{h}", help="基準分數仍來自最近完整日線；即時價量只按週期權重重排。")
            if use_live:
                intraday_fragment(snap,h)
            else:
                _render_daily_list(snap,h,5)
        else:
            _render_daily_list(snap,h,5)
    else:
        st.markdown("<div class='section-kicker'>CHECK</div><div class='section-title'>個股診斷</div>", unsafe_allow_html=True)
        _doctor(snap)

    st.caption("歷史案例用於比較，不代表未來結果；V16.1 以完整日線做基準，盤中總覽與各週期自動套用即時行情重排；市場主線採官方產業＋供應鏈雙層 breadth，配置與訊號分離。")


if __name__ == "__main__":
    main()
