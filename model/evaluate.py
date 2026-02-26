#!/usr/bin/env python3

import csv
from pathlib import Path


DATASET = Path(__file__).resolve().parent.parent / "data" / "TSLA.csv"


def to_float(value: str) -> float:
    return float((value or "0").replace(",", "").strip())


def load_rows():
    with DATASET.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    print("=" * 68)
    print("TESLA FORECAST EVALUATION")
    print("=" * 68)

    if not DATASET.exists():
        print(f"Dataset missing: {DATASET}")
        return

    rows = load_rows()
    closes = [to_float(r["Close"]) for r in rows]
    if len(closes) < 24:
        print("Need at least 24 rows for evaluation")
        return

    lookback = 12
    correct = 0
    total = 0

    for i in range(lookback, len(closes) - 1):
        history = closes[i - lookback : i]
        last_close = closes[i]
        next_close = closes[i + 1]

        avg_ret = 0.0
        ret_count = 0
        for j in range(1, len(history)):
            prev = history[j - 1]
            curr = history[j]
            if prev > 0:
                avg_ret += (curr - prev) / prev
                ret_count += 1

        if ret_count == 0:
            continue
        avg_ret /= ret_count

        pred_up = avg_ret >= 0
        actual_up = next_close >= last_close

        if pred_up == actual_up:
            correct += 1
        total += 1

    if total == 0:
        print("No valid evaluation windows")
        return

    accuracy = correct / total
    print(f"Samples evaluated: {total}")
    print(f"Directional accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
