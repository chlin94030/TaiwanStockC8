"""Import app.py without a real Streamlit runtime; useful for CI syntax/API smoke."""
from __future__ import annotations
import sys
import types
import numpy as np
import pandas as pd

st = types.ModuleType("streamlit")
def fragment(*args, **kwargs):
    def deco(fn):
        return fn
    return deco
st.fragment = fragment
sys.modules.setdefault("streamlit", st)

import app  # noqa: E402

assert app.VIEWS[0] == "總覽"
assert set(app.H_LABEL) == {"short", "mid", "long"}

# Technical dataframe smoke: the final UI must carry all requested indicators.
dates = pd.bdate_range("2025-09-01", periods=260)
close = np.linspace(100.0, 160.0, len(dates)) + np.sin(np.arange(len(dates))/8.0)*2.0
ohlcv = np.column_stack([close-0.5, close+1.0, close-1.0, close, np.linspace(1_000_000, 1_800_000, len(dates))])
chart = {"dates": dates.strftime("%Y-%m-%d").tolist(), "ohlcv": ohlcv.tolist()}
df = app._chart_dataframe(chart)
for col in ["MA5","MA20","MA60","MA120","MA240","BBU","BBL","KD_K","KD_D","MACD","MACD_SIGNAL","MACD_HIST"]:
    assert col in df.columns, col
assert np.isfinite(df["MA240"].iloc[-1])
assert np.isfinite(df["KD_K"].iloc[-1])
assert np.isfinite(df["MACD"].iloc[-1])

print("Alpha Radar V16 app import + technical helper smoke: PASS")
