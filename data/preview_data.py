#!/usr/bin/env python3

import csv
from pathlib import Path


DATASET = Path(__file__).with_name("TSLA.csv")


def to_float(value: str) -> float:
    return float((value or "0").replace(",", "").strip())


def load_rows():
    with DATASET.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    print("=" * 72)
    print("TSLA DATA PREVIEW")
    print("=" * 72)

    if not DATASET.exists():
        print(f"Dataset not found: {DATASET}")
        return

    rows = load_rows()
    if not rows:
        print("Dataset is empty")
        return

    closes = [to_float(r["Close"]) for r in rows if r.get("Close")]
    volumes = [to_float(r["Volume"]) for r in rows if r.get("Volume")]

    returns = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        curr = closes[i]
        if prev > 0:
            returns.append((curr - prev) / prev)

    print(f"Rows: {len(rows):,}")
    print(f"Date range: {rows[0]['Date']} -> {rows[-1]['Date']}")
    print(f"Latest close: {closes[-1]:.2f}")
    print(f"Average close: {sum(closes) / len(closes):.2f}")
    print(f"Average volume: {sum(volumes) / len(volumes):,.0f}")
    if returns:
        avg_ret = sum(returns) / len(returns)
        print(f"Average daily return: {avg_ret * 100:.3f}%")

    print("\nSample rows:")
    for row in rows[-3:]:
        print(
            f"  {row['Date']}  O:{to_float(row['Open']):.2f}  H:{to_float(row['High']):.2f}  "
            f"L:{to_float(row['Low']):.2f}  C:{to_float(row['Close']):.2f}  V:{to_float(row['Volume']):,.0f}"
        )


if __name__ == "__main__":
    main()
