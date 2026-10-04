# Alpha Radar V16 驗效狀態

## 已完成（本環境）

- 全 Python compileall：PASS
- Offline self-test：PASS
- App import + technical helper smoke：PASS
- 100 組長歷史合成壓力測試：PASS
- benchmark-aligned / de-overlapped historical analog：PASS
- anti-chase / entry plan：PASS
- 公司 Evidence Gate / red-flag：PASS
- 官方營收/EPS/毛利/PER fallback：PASS
- official-sector breadth：PASS
- supply-chain breadth：PASS
- two-layer mainline resonance：PASS
- 熱門主線不能掩蓋基本面惡化：PASS
- intraday blend / stale quote rejection：PASS
- branch top-buy/top-sell parser：PASS
- holding-distribution parser：PASS
- scan lock / atomic snapshot / SQLite production helper：PASS

## 真實 100 檔 Walk-forward

V16 `validate_100.py` 的 price-core 會在每個歷史 cutoff 使用**當時以前**的：
- 個股價格
- TAIEX
- official-sector breadth
- supply-chain breadth

重新做同日 cross-sectional ranking，再觀察之後 10/40/120 交易日 realized return。這避免拿今天的市場主線回填歷史。

本容器實際執行後回報 `NOT_RUN`，原因是外部網路不可用，股票母體僅取得 8 檔安全備援；沒有以合成資料替代 100 檔真實驗效。

上傳 GitHub 後請執行：

`Actions → Validate 100 real Taiwan stocks → Run workflow`

## 尚未覆蓋的驗效範圍

完整歷史基本面、法人與券商分點尚未建立 point-in-time archive，因此 100 檔 validator **不把這些現值回填歷史**。這是刻意避免 look-ahead，不是漏測後假裝已驗證。

## 判定

V16 可視為「程式/資料流/模型結構已完成回歸測試的 production candidate」，但不能稱為完美模型或保證勝率。真正績效判定仍需 networked 100-stock walk-forward 與後續 out-of-sample 實盤觀察。
