#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


@dataclass
class MatchupResult:
    weight_class: str
    fighter_a: str
    fighter_b: str
    a_odds: str
    b_odds: str
    status: str


def parse_percent(value: str) -> float | None:
    if not value or not value.endswith("%"):
        return None
    try:
        return float(value[:-1])
    except ValueError:
        return None


def parse_pairs(values: Sequence[str]) -> List[Tuple[str, str, str]]:
    pairs: List[Tuple[str, str, str]] = []
    for value in values:
        parts = [part.strip() for part in value.split("|")]
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"Invalid --pair format: {value}. Use 'Weight Class|Fighter A|Fighter B'.")
        pairs.append((parts[0], parts[1], parts[2]))
    return pairs


def prompt_pairs() -> List[Tuple[str, str, str]]:
    print("Enter matchups to generate odds rows.")
    print("Type 'q' for weight class to finish.\n")

    pairs: List[Tuple[str, str, str]] = []
    while True:
        weight_class = input("Weight class: ").strip()
        if weight_class.lower() == "q":
            break
        if not weight_class:
            print("Weight class is required.\n")
            continue

        fighter_a = input("Fighter A name: ").strip()
        fighter_b = input("Fighter B name: ").strip()
        if not fighter_a or not fighter_b:
            print("Both fighter names are required.\n")
            continue

        pairs.append((weight_class, fighter_a, fighter_b))
        print("")

    return pairs


def run_matchup(model_dir: Path, weight_class: str, fighter_a: str, fighter_b: str) -> MatchupResult:
    if not (model_dir / "ufc_nn").exists():
        return MatchupResult(weight_class, fighter_a, fighter_b, "N/A", "N/A", "model executable not found")

    proc = subprocess.run(
        ["./ufc_nn", "--load", "--matchup"],
        input=f"{weight_class}\n{fighter_a}\n{fighter_b}\n",
        cwd=str(model_dir),
        capture_output=True,
        text=True,
        check=False,
    )

    output = f"{proc.stdout}\n{proc.stderr}" if proc.stderr else proc.stdout

    if "Class model not found" in output:
        return MatchupResult(weight_class, fighter_a, fighter_b, "N/A", "N/A", "class model missing")
    if "No stats found for fighter A" in output or "No stats found for fighter B" in output:
        return MatchupResult(weight_class, fighter_a, fighter_b, "N/A", "N/A", "fighter stats missing")
    if "Invalid weight class input" in output:
        return MatchupResult(weight_class, fighter_a, fighter_b, "N/A", "N/A", "invalid weight class")

    probs = re.findall(r"^P\((.+?) wins\)\s*:\s*([0-9]+(?:\.[0-9]+)?)%$", output, flags=re.MULTILINE)

    if len(probs) < 2:
        status = "prediction unavailable"
        if proc.returncode != 0:
            status = f"model exit code {proc.returncode}"
        return MatchupResult(weight_class, fighter_a, fighter_b, "N/A", "N/A", status)

    parsed = {name.strip(): value.strip() + "%" for name, value in probs}
    a_odds = parsed.get(fighter_a, "N/A")
    b_odds = parsed.get(fighter_b, "N/A")

    if a_odds == "N/A" or b_odds == "N/A":
        return MatchupResult(weight_class, fighter_a, fighter_b, a_odds, b_odds, "name mismatch in model output")

    return MatchupResult(weight_class, fighter_a, fighter_b, a_odds, b_odds, "ok")


def format_table(rows: Iterable[MatchupResult]) -> str:
    rows_list = list(rows)
    headers = ["Weight Class", "Fighter A", "Fighter B", "A Odds", "B Odds", "Status"]

    data_rows = [
        [
            row.weight_class,
            row.fighter_a,
            row.fighter_b,
            row.a_odds,
            row.b_odds,
            row.status,
        ]
        for row in rows_list
    ]

    widths = [len(header) for header in headers]
    for data in data_rows:
        for idx, value in enumerate(data):
            widths[idx] = max(widths[idx], len(value))

    def render_line(values: Sequence[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    header = render_line(headers)
    separator = "-+-".join("-" * width for width in widths)
    lines = [header, separator]
    lines.extend(render_line(data) for data in data_rows)
    return "\n".join(lines)


def write_output(path: Path, rows: List[MatchupResult]) -> None:
    title = "UFC Odds Table"
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body = format_table(rows)
    content = f"{title}\nGenerated: {generated_at}\n\n{body}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate an odds table for fighter matchups.")
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Matchup in format 'Weight Class|Fighter A|Fighter B'. Repeat for multiple rows.",
    )
    parser.add_argument(
        "--out",
        default=str(root / "odds" / "odds.txt"),
        help="Output file path for the table (default: odds/odds.txt).",
    )
    parser.add_argument(
        "--model-dir",
        default=str(root / "model"),
        help="Path to model directory containing ufc_nn.",
    )
    parser.add_argument(
        "--min-odds",
        type=float,
        default=75.0,
        help="Only write rows where at least one fighter has odds >= this value (default: 75).",
    )
    args = parser.parse_args()

    pairs = parse_pairs(args.pair)
    if not pairs:
        pairs = prompt_pairs()

    if not pairs:
        print("No matchups entered. Nothing written.")
        return

    model_dir = Path(args.model_dir).resolve()
    output_path = Path(args.out).resolve()

    results: List[MatchupResult] = []
    skipped_low_odds = 0
    for weight_class, fighter_a, fighter_b in pairs:
        result = run_matchup(model_dir, weight_class, fighter_a, fighter_b)

        has_missing_odds = result.a_odds == "N/A" or result.b_odds == "N/A"
        has_missing_status = result.status == "fighter stats missing"
        if has_missing_odds or has_missing_status:
            print(
                f"Processed: {fighter_a} vs {fighter_b} [{weight_class}] -> skipped (missing fighter stats/odds)"
            )
            continue

        a_pct = parse_percent(result.a_odds)
        b_pct = parse_percent(result.b_odds)

        both_below_threshold = (
            a_pct is not None
            and b_pct is not None
            and a_pct < args.min_odds
            and b_pct < args.min_odds
        )

        if both_below_threshold:
            skipped_low_odds += 1
            print(
                f"Processed: {fighter_a} vs {fighter_b} [{weight_class}] -> skipped (both odds below {args.min_odds:.2f}%)"
            )
            continue

        results.append(result)
        print(f"Processed: {fighter_a} vs {fighter_b} [{weight_class}] -> {result.status}")

    write_output(output_path, results)
    if skipped_low_odds:
        print(f"Skipped {skipped_low_odds} matchup(s) below min odds threshold.")
    print(f"\nWrote odds table to: {output_path}")


if __name__ == "__main__":
    main()
