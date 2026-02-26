#!/usr/bin/env python3

import csv
import json
from pathlib import Path


DATASET = Path(__file__).with_name("TSLA.csv")
OUT = Path(__file__).with_name("tesla_feature_snapshot.json")


def to_float(value: str) -> float:
    return float((value or "0").replace(",", "").strip())


def moving_average(values, window: int) -> float:
    if len(values) < window:
        return sum(values) / len(values)
    subset = values[-window:]
    return sum(subset) / len(subset)


def main() -> None:
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    with DATASET.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(rows) < 3:
        raise RuntimeError("Need at least 3 rows to compute snapshot")

    closes = [to_float(r["Close"]) for r in rows]
    volumes = [to_float(r["Volume"]) for r in rows]

    daily_returns = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        curr = closes[i]
        if prev > 0:
            daily_returns.append((curr - prev) / prev)

    latest = rows[-1]
    previous = rows[-2]

    snapshot = {
        "symbol": "TSLA",
        "latest_date": latest["Date"],
        "latest_close": closes[-1],
        "previous_close": closes[-2],
        "last_return_pct": ((closes[-1] - closes[-2]) / closes[-2]) * 100.0,
        "ma_3": moving_average(closes, 3),
        "ma_6": moving_average(closes, 6),
        "ma_12": moving_average(closes, 12),
        "avg_volume_6": moving_average(volumes, 6),
        "avg_return_pct": (sum(daily_returns) / len(daily_returns)) * 100.0,
        "latest_open": to_float(latest["Open"]),
        "latest_high": to_float(latest["High"]),
        "latest_low": to_float(latest["Low"]),
        "previous_date": previous["Date"],
    }

    OUT.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"Wrote Tesla feature snapshot to {OUT}")


if __name__ == "__main__":
    main()
