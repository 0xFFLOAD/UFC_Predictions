#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass
class ForecastRow:
    date: str
    time: str
    predicted_close: str
    predicted_return: str
    bullish_probability: str
    direction: str
    status: str


def to_military_time(value: str) -> str:
    match = re.match(r"^(\d{1,2}):(\d{2})$", value)
    if not match:
        return value
    return f"{int(match.group(1)):02d}:{int(match.group(2)):02d}"


def parse_date_inputs(values: Sequence[str]) -> List[tuple[str, str]]:
    out: List[tuple[str, str]] = []
    for value in values:
        value = value.strip()
        if "T" in value:
            date_part, time_part = value.split("T", 1)
            out.append((date_part, to_military_time(time_part)))
        else:
            out.append((value, "00:00"))
    return out


def prompt_dates() -> List[tuple[str, str]]:
    rows: List[tuple[str, str]] = []
    print("Enter forecast dates. Type 'q' to finish.")
    while True:
        date_value = input("Date (YYYY-MM-DD): ").strip()
        if date_value.lower() == "q":
            break
        if not date_value:
            continue
        time_value = input("Time (HH:MM, default 00:00): ").strip() or "00:00"
        rows.append((date_value, to_military_time(time_value)))
    return rows


def run_forecast(model_dir: Path, forecast_date: str) -> ForecastRow:
    binary = model_dir / "tsla_nn"
    if not binary.exists():
        return ForecastRow(forecast_date, "00:00", "N/A", "N/A", "N/A", "N/A", "model executable not found")

    proc = subprocess.run(
        ["./tsla_nn", "--load", "--predict", "--date", forecast_date],
        cwd=str(model_dir),
        text=True,
        capture_output=True,
        check=False,
    )

    output = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")

    close_match = re.search(r"Predicted close\s*:\s*([0-9]+(?:\.[0-9]+)?)", output)
    return_match = re.search(r"Expected return\s*:\s*([\-0-9]+(?:\.[0-9]+)?)%", output)
    prob_match = re.search(r"Bullish probability\s*:\s*([0-9]+(?:\.[0-9]+)?)%", output)

    if not close_match or not return_match or not prob_match:
        status = "forecast unavailable" if proc.returncode == 0 else f"model exit code {proc.returncode}"
        return ForecastRow(forecast_date, "00:00", "N/A", "N/A", "N/A", "N/A", status)

    predicted_close = float(close_match.group(1))
    predicted_return = float(return_match.group(1))
    bullish_prob = float(prob_match.group(1))
    direction = "UP" if predicted_return >= 0 else "DOWN"

    return ForecastRow(
        forecast_date,
        "00:00",
        f"{predicted_close:.2f}",
        f"{predicted_return:.2f}%",
        f"{bullish_prob:.2f}%",
        direction,
        "ok",
    )


def format_table(rows: Iterable[ForecastRow]) -> str:
    rows_list = list(rows)
    headers = ["Date", "Time", "Predicted Close", "Predicted Return", "Bullish Prob", "Direction", "Status"]
    data = [
        [r.date, r.time, r.predicted_close, r.predicted_return, r.bullish_probability, r.direction, r.status]
        for r in rows_list
    ]

    widths = [len(h) for h in headers]
    for row in data:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def render(parts: Sequence[str]) -> str:
        return " | ".join(parts[i].ljust(widths[i]) for i in range(len(parts)))

    lines = [render(headers), "-+-".join("-" * w for w in widths)]
    lines.extend(render(row) for row in data)
    return "\n".join(lines)


def read_existing_rows(path: Path) -> List[ForecastRow]:
    if not path.exists():
        return []

    rows: List[ForecastRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "|" not in line:
            continue
        if line.strip().startswith("Date") and "Predicted Close" in line:
            continue
        if set(line.replace("|", "").replace("-", "").replace("+", "").strip()) == set():
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 7:
            continue
        rows.append(ForecastRow(*parts))
    return rows


def write_output(path: Path, rows: List[ForecastRow]) -> None:
    title = "TSLA Forecast Table"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{title}\nGenerated: {timestamp}\n\n{format_table(rows)}\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate TSLA forecast table.")
    parser.add_argument("--date", action="append", default=[], help="Forecast date: YYYY-MM-DD or YYYY-MM-DDTHH:MM")
    parser.add_argument("--out", default=str(root / "odds" / "odds.txt"), help="Output table path")
    parser.add_argument("--model-dir", default=str(root / "model"), help="Model directory")
    parser.add_argument("--min-confidence", type=float, default=55.0, help="Minimum bullish probability distance from 50 to keep row")
    args = parser.parse_args()

    date_inputs = parse_date_inputs(args.date)
    if not date_inputs:
        date_inputs = prompt_dates()
    if not date_inputs:
        print("No forecast dates provided.")
        return

    model_dir = Path(args.model_dir).resolve()
    output_path = Path(args.out).resolve()

    binary = model_dir / "tsla_nn"
    if not binary.exists():
        print("Model binary missing, attempting build...")
        subprocess.run(["make"], cwd=str(model_dir), check=False)

    model_bin = model_dir / "tsla_model.bin"
    if not model_bin.exists():
        print("Model file missing, training baseline...")
        subprocess.run(["./tsla_nn", "--train"], cwd=str(model_dir), check=False)

    new_rows: List[ForecastRow] = []
    skipped_low_conf = 0
    for date_value, time_value in date_inputs:
        row = run_forecast(model_dir, date_value)
        row.time = to_military_time(time_value)
        if row.status != "ok":
            print(f"Processed: {date_value} {row.time} -> skipped ({row.status})")
            continue

        prob = float(row.bullish_probability.rstrip("%"))
        confidence = max(prob, 100.0 - prob)
        if confidence < args.min_confidence:
            skipped_low_conf += 1
            print(f"Processed: {date_value} {row.time} -> skipped (confidence {confidence:.2f}% < {args.min_confidence:.2f}%)")
            continue

        new_rows.append(row)
        print(f"Processed: {date_value} {row.time} -> ok")

    existing_rows = read_existing_rows(output_path)
    write_output(output_path, existing_rows + new_rows)

    if skipped_low_conf:
        print(f"Skipped {skipped_low_conf} forecast(s) below confidence threshold.")
    if existing_rows:
        print(f"Appended {len(new_rows)} row(s) to {len(existing_rows)} existing row(s).")
    print(f"\nWrote forecast table to: {output_path}")


if __name__ == "__main__":
    main()
