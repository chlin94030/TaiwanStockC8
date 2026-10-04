# V16.0 Release Notes — Final Evidence

- 產業主線從單層 broad industry 升級為「官方產業 56% + 供應鏈 breadth 44%」。
- 新增供應鏈 bucket：半導體廠務/設備、封裝測試、PCB/高速材料、AI伺服器/散熱、工業電腦等；breadth 動態計算，不硬編碼熱門股加分。
- 100 檔 walk-forward validator 在每個歷史 cutoff 同時計算當時 official-sector + supply-chain breadth，避免 look-ahead。
- 盤中頁擴展到短/中/長三週期；顯示基準完整日線日期與即時抓取日期時間。
- 開盤期間更新基準模型時保留最近完整收盤 K；即時價格只做 overlay，避免 partial daily bar 污染 MA/KD/MACD。
- K 線改為 category 交易日軸，週末/休市不留大空白；K 棒間距縮小。
- 1m/3m/6m/1y 增加雙端日期縮放 slider，可穩定放大任意交易日區間，不使用容易誤觸的手機拖圖 pan。
- 個股研究新增 24 月營收圖、8 季 EPS 圖、近月法人日別。
- 有 FinMind 權限時，可按需載入近 10 日券商分點：前三大淨買/淨賣、日別分點淨額；不在全市場掃描時批次強抓。
- 有 Backer/Sponsor 權限時，可按需載入大戶/散戶持股近似比例與趨勢。
- 「主力分點代理未啟用」改為「可載入」，避免誤解成系統不支援。
- 修正力成細產業標籤為記憶體/邏輯封裝測試。
- 沿用 V14/V15 production guardrails：scan lock、atomic snapshot、SQLite WAL、增量行情、官方 fallback、Evidence Gate、anti-chase、Signal/Allocation 分離。
