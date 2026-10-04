# ALPHA/TW FINAL EVIDENCE 16.0

V16 是目前的收斂版：**完整日線基準模型 + 盤中即時覆蓋 + 官方產業/供應鏈雙層主線 + 公司證據 + 法人/選配籌碼**。設計目標不是追求某一份當下名單，而是降低「題材追高、公司資料缺失、跨週期誤判、盤中半根 K 污染長期指標」四類常見錯誤。

## 核心結構

1. **完整日線基準**：MA5/20/60/120/240、ATR、Bollinger、KD、MACD、相對大盤、歷史相似案例只使用已完成交易日。
2. **盤中 overlay**：09:00–13:30 若設定 FinMind Sponsor Token，約每 60 秒重新抓取即時快照並重排候選池；短/中/長的盤中權重分別為 45% / 15% / 5%。未收盤的半根日 K 不放入長期模型。
3. **雙層市場主線**：官方 broad industry breadth + 穩定的供應鏈 theme breadth。主線只是一層因子，不能抵消負 EPS、負毛利或持續營收衰退。
4. **Evidence Complete**：營收、EPS/毛利、估值、法人優先使用 FinMind 歷史；缺資料時以 TWSE/TPEx 官方最新期資料補位。完整資料模式另有中長線 evidence gate。
5. **Signal / Allocation 分離**：同一強股可以真實呈現跨週期共識；實際配置 allocator 再做重複與產業集中控制。

## 模型權重

| 因子 | 短線 | 中線 | 長線 |
|---|---:|---:|---:|
| 個股技術 | 36% | 27% | 20% |
| 歷史相似案例 | 17% | 18% | 18% |
| 公司基本面 | 7% | 27% | 43% |
| 法人籌碼 | 18% | 10% | 5% |
| 流動性 | 7% | 4% | 3% |
| 市場主線（官方產業 + 供應鏈） | 15% | 14% | 11% |

缺資料時按現有證據重新正規化；中長線仍受 Evidence Gate 與 fundamental red-flag 限制。

## 盤中更新怎麼運作

- **開盤期間按「更新基準模型」**：保留最近完整收盤日作為基準，更新可安全取得的歷史/公司資料，不把盤中尚未完成的日 K 混進 MA/KD/MACD。
- **即時頁或短/中/長頁的「盤中即時覆蓋」**：使用即時價、相對大盤、盤中均價、量比、日內位置、買賣盤等重排；畫面同時顯示「基準日線日期」與「即時抓取日期時間」。
- **收盤後**：官方完整 EOD 可用後再成為新的日線基準。

FinMind 台股即時快照屬 Sponsor 權限，來源本身約 10 秒更新；本 App 預設 60 秒重排以兼顧穩定與 API 負擔。

## 個股頁面

- 月營收：最多 24 個月歷史圖，並顯示 MoM / YoY。
- 財務：近 8 季 EPS 圖；主要卡仍使用近 4 季/官方最新累計 EPS 與毛利率。
- 籌碼：近月外資、投信、自營商日別表。
- 券商分點：**按需載入**近 10 個交易日，顯示前三大淨買/淨賣分點及日別分點淨額；這只是交易行為代理，不是官方「主力」分類。
- 股權分布：按需載入大戶（約 1000 張以上）與散戶（約 400 張以下）持股比例及趨勢，只作輔助觀察，不進核心排名。
- K 線：日 K + 週/月/季/半年/年線（MA5/20/60/120/240）+ Bollinger + 量能 + KD + MACD。交易日使用 category 軸，不為週末/休市留下大空白；1m/3m/6m/1y 預設區間可用雙端日期 slider 精準縮放。

## 為什麼分點/股權資料不在全市場掃描時一起抓

分點資料權限較高、量大且盤後更新。若替 1,000 檔候選一次抓完，會顯著拉長更新時間並增加失敗率。V16 因此把它設為個股展開後的 **on-demand evidence**：需要時才抓，不讓輔助功能拖垮核心模型。

## Streamlit Secrets

在 Streamlit Cloud 的 Secrets 設定：

```toml
FINMIND_TOKEN = "你的 token"
```

`ENABLE_BRANCH_FLOW = "1"` 只在你確定要讓掃描程序自動批次補分點時才設定；一般使用不需要，個股頁按鈕會在有 Token 時按需載入。

> 若 App 對外公開，請自行確認資料供應商授權。FinMind 官方授權頁目前明載：即時原始資料不允許直接展示於公開 web/app；本專案預設適合作為個人/內部研究工具。

## 部署

```bash
python -m pip install -r requirements.txt
python healthcheck.py
python self_test.py
python smoke_app_import.py
streamlit run app.py
```

GitHub Actions 另有：
- `Alpha Radar CI`：Python 3.11 / 3.12 compile + self-test + healthcheck + app smoke。
- `Validate 100 real Taiwan stocks`：固定 seed 的 100 檔真實台股 cross-sectional walk-forward。

## 驗效邊界

V16 的 offline regression、100 組長歷史壓力測試、技術/產業/證據/盤中/營運保護已通過；但**不能宣稱模型已「完美」或保證勝率**。本環境沒有完整外網/yfinance，因此 100 檔真實資料驗效在此主動回報 NOT_RUN；上傳 GitHub 後請執行 validation workflow。該 validator 對每個歷史 cutoff 重新計算當時可得的價格、大盤、官方產業與供應鏈 breadth，不用今天的主線回填歷史；歷史財報/法人目前仍未做完整 point-in-time backfill，因此不把它們偽裝成已驗證績效。
