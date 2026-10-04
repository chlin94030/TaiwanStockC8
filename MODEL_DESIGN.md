# Alpha Radar V16 — Final Evidence 模型設計

## 1. 目標

V16 將選股拆成五層：**Market Regime → Official Sector Breadth → Supply-chain Breadth → Stock Evidence → Execution / Intraday Overlay**。目的不是把近期熱門名稱寫死，而是辨識「整個資金主線是否真的有廣度」，再要求個股自己的技術、基本面與法人證據成立。

## 2. 完整日線原則

所有中長期技術指標與 historical analog 只使用完整交易日。盤中 10:30 的價格不被當成當日收盤，不參與 MA20/60/120/240、KD、MACD 或歷史相似案例，避免 partial-bar bias。

## 3. Market Regime

TAIEX 以 MA20 / MA60 與價格位置區分 Bull / Neutral / Bear。Regime 主要調節短中線部位節奏，不會在 Bear 自動清空所有股票。

## 4. 雙層 Mainline

### 4.1 Official sector breadth
官方 broad industry 依同業股票計算：
- median 20 日相對 TAIEX 報酬
- median 60 日相對 TAIEX 報酬
- 站上 MA20 比例
- 站上 MA60 比例
- 近期量能參與比例

### 4.2 Supply-chain breadth
官方產業太粗，例如「其他電子」可能同時含半導體廠務、EMS、不同題材。V16 因此加入穩定供應鏈 bucket（半導體廠務/設備、封裝測試、PCB/高速材料、AI伺服器散熱、工業電腦等），再用**相同 breadth/RS 算法**計分。

這不是「半導體固定加分」。供應鏈只有當多檔成分股同時領先才會變成主線；若未來資金轉向金融、航運、重電，breadth 會自行換檔。

### 4.3 Combined mainline
主線分數約為：官方產業 56% + 供應鏈 44%。最終主線權重只占：短 15%、中 14%、長 11%。

## 5. Stock Evidence

| 因子 | 短線 | 中線 | 長線 |
|---|---:|---:|---:|
| technical | 36 | 27 | 20 |
| historical analog | 17 | 18 | 18 |
| fundamental | 7 | 27 | 43 |
| institutional flow | 18 | 10 | 5 |
| liquidity | 7 | 4 | 3 |
| combined mainline | 15 | 14 | 11 |

基本面不是單月 YoY：使用近期 3/6 月營收持續性、EPS 正向比例/加速度、毛利品質、成長調整估值及 explicit red flags。金融業採不同公司證據組合。

## 6. Mainline Resonance

只有「主線強 + 個股自身證據強」才有額外 0–3 分共振：
- 短線：技術與法人/量價確認。
- 中線：技術 + 營運持續性。
- 長線：基本面品質優先。

熱產業不能抵銷負 EPS、負毛利與長期營收衰退。此項有 regression test。

## 7. 歷史相似案例

- 每個歷史日期使用當時 TAIEX 特徵，而非今天的大盤狀態。
- 相鄰歷史日期去重，降低同一波行情重複計數。
- 以 10 / 40 / 120 交易日後的實際報酬形成分布，不以近期 drift 直接複利外推。

## 8. 盤中 Overlay

基準仍是完整日線排名。FinMind realtime snapshot 只重排已通過基準模型的候選池：
- short：45% intraday + 55% daily
- mid：15% intraday + 85% daily
- long：5% intraday + 95% daily

盤中訊號包含相對大盤、VWAP/均價差、量比、開盤後強弱、日內位置、買賣盤壓力、成交金額與過熱/價差扣分。盤中急拉不能推翻長線公司品質。

## 9. 籌碼資料

核心排名使用三大法人月內淨買賣。券商分點與股權分布屬選配輔助：
- 分點：近 10 個交易日、前三大淨買/淨賣及日別淨額，需 Sponsor 權限，通常約 21:00 更新。
- 股權：大戶/散戶持股分級，需 Backer/Sponsor；只作觀察，不直接進排名。

這樣可避免資料權限或分點 API 波動造成整個市場掃描失敗。

## 10. Signal vs Allocation

各週期 signal list 保留真實共識：一檔股票若短/中/長真的都強，可以 3/3 出現。Portfolio allocator 才控制同一持股與產業集中。這避免為了「去重」把次佳股票硬塞進錯週期。

## 11. 模型不能聲稱完美的理由

- 市場 regime、主線與公司狀況會改變。
- 任何歷史回測都有 sample / survivorship / data-availability 風險。
- 目前完整 100-stock validator 驗證的是 price + benchmark + point-in-time two-layer mainline core；歷史基本面與法人尚未全面 point-in-time backfill。
- 所以 V16 的目標是降低結構性錯誤與資料洩漏，而不是宣稱固定勝率。
