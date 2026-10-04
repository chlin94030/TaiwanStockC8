# Alpha Radar V16 Test Report

## 本次測試目的

V16 是收斂版，測試不只確認「程式可以 import」，而是針對本次新增的雙層主線、盤中/盤後分離、K線技術指標、日期縮放、分點/股權選配資料與既有 production guardrails 做回歸。

## 本環境實際執行

### 1. 全 Python 編譯

```bash
python -m compileall -q .
```

結果：**PASS**。

### 2. Offline regression / stress

```bash
python self_test.py
```

結果：`Alpha Radar V16 offline self-test: PASS`。

涵蓋：
- 10/40/120 日 historical analog 與 benchmark alignment
- 100 組不同參數長歷史壓力測試
- anti-chase / entry plan
- 基本面持續性、EPS/毛利、估值紅旗
- 三大法人解析
- 官方營收/EPS/毛利/PER fallback
- 中長線 Evidence Gate
- official sector breadth
- supply-chain theme breadth
- 熱門主線不能救負 EPS/負毛利公司
- signal list / portfolio allocation 分離
- 盤中 rerank 與 overheat penalty
- scan lock / stale lock recovery / atomic snapshot
- 大戶/散戶 HoldingSharesLevel parser
- 分點前三大淨買/淨賣及 daily_rows parser

### 3. App import + technical helper smoke

```bash
python smoke_app_import.py
```

結果：`Alpha Radar V16 app import + technical helper smoke: PASS`。

另外以 260 個交易日合成 K 線確認：
`MA5/20/60/120/240`, `BBU/BBL`, `KD_K/KD_D`, `MACD/MACD_SIGNAL/MACD_HIST` 全部存在且最新值有限。

### 4. Deployment healthcheck

```bash
python healthcheck.py
```

本封裝容器結果：**DEPENDENCY_FAIL**，原因為容器沒有安裝 `streamlit` 與 `yfinance`。這是環境限制而非模型測試失敗；本環境亦無法從 PyPI 安裝外部套件。GitHub CI / Streamlit Cloud 會先 `pip install -r requirements.txt` 再執行 healthcheck。

### 5. 100 檔真實台股 validation 實際嘗試

```bash
python validate_100.py --stocks 100 --period 8y --points 6
```

結果：**NOT_RUN**。外部網路不可用時，TWSE/TPEx universe 只能回到 8 檔內建安全備援，因此 validator 主動中止：

`股票名單僅取得 8 檔，無法做 100 檔真實標的驗證`

沒有用合成股票冒充真實績效。

## V16 特別回歸

1. **雙層主線**：官方產業與供應鏈 breadth 分別計算；強群聚 > 弱群聚，且非人工熱門名單加分。
2. **公司紅旗優先**：高主線分數但營收崩落、EPS/毛利為負的公司，長線分數必須明顯低於基本面持續改善的公司。
3. **盤中不污染日線**：intraday engine 只 blend ranking，不改寫基準 OHLC/MA/KD/MACD。
4. **盤中時間戳**：RealtimeStatus 會帶 `fetched_at`，UI 同時顯示基準完整日線日期與即時抓取時間。
5. **K線交易日等距**：Plotly x-axis 使用 `category`；週末/休市沒有額外空白。
6. **穩定縮放**：用 Streamlit 雙端日期 `select_slider` 切片資料，不用手機 pan/scrollZoom；改動端點不會把整張圖拖離座標。
7. **分點資料失敗隔離**：無 Token / 無 Sponsor 權限只讓按需籌碼區顯示 unavailable，不影響主模型與快照。
8. **股權資料失敗隔離**：同上，holding distribution 只作 display-only evidence。

## 未聲稱通過的項目

- 尚未在本環境完成有網路的 100 檔真實台股 8 年 cross-sectional walk-forward。
- 尚未對歷史基本面與法人做完整 point-in-time backfill；因此不能把今日可用的財報/法人資料回填到歷史並宣稱全模型勝率。
- 沒有任何測試可以證明未來報酬或「完美模型」。V16 的測試目標是降低程式錯誤、look-ahead、資料缺失與結構性選股偏誤。
