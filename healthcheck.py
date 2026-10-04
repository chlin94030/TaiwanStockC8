"""Alpha Radar V16 deployment smoke check.

Run after installing requirements:
    python healthcheck.py
No live network is required unless --live is passed.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import tempfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also verify public universe endpoints")
    args = ap.parse_args()

    required = ["numpy", "pandas", "plotly", "requests", "streamlit", "yfinance"]
    missing = []
    for name in required:
        try:
            importlib.import_module(name)
        except Exception as exc:
            missing.append(f"{name}: {exc}")
    if missing:
        print("DEPENDENCY_FAIL")
        for x in missing:
            print(" -", x)
        return 2

    import radar_service
    from market_data import DailyPriceStore, fetch_twse_universe
    from operational_tools import atomic_write_json

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        store = DailyPriceStore(root / "prices.sqlite")
        assert store.get_prices("2330.TW").empty
        target = root / "snapshot.json"
        atomic_write_json(target, {"model_version": radar_service.OPERATIONS_VERSION, "ok": True})
        assert target.exists() and '"ok":true' in target.read_text(encoding="utf-8")

    if args.live:
        u = fetch_twse_universe()
        if len(u) < radar_service.MIN_PRODUCTION_UNIVERSE:
            print(f"LIVE_FAIL universe={len(u)}")
            return 3
        print(f"LIVE_OK universe={len(u)}")

    print("Alpha Radar V16 healthcheck: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
