#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path

DATASET = Path(__file__).with_name("ufc_fights_full_with_odds.csv")
OUT = Path(__file__).with_name("fighter_stats_dict.json")
PRIOR_WEIGHT = 6.0

FIELDS = [
    "height",
    "reach",
    "age",
    "sig_str_pm",
    "sig_acc",
    "sig_abs",
    "sig_def",
    "td_avg",
    "td_acc",
    "td_def",
    "sub_avg",
    "weight",
]


def to_float(value: str):
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def add_record(agg, key, values):
    bucket = agg[key]
    for field, val in zip(FIELDS, values):
        if val is None:
            continue
        bucket[field][0] += val
        bucket[field][1] += 1


def add_class_record(class_agg, weight_class, values):
    bucket = class_agg[weight_class]
    for field, val in zip(FIELDS, values):
        if val is None:
            continue
        bucket[field][0] += val
        bucket[field][1] += 1


def main():
    agg = defaultdict(lambda: {f: [0.0, 0] for f in FIELDS})
    class_agg = defaultdict(lambda: {f: [0.0, 0] for f in FIELDS})

    with DATASET.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wc = (row.get("weight_class") or "").strip()
            if wc == "":
                continue

            f1 = (row.get("fighter_a_name") or "").strip()
            if f1:
                f1_sig_landed = to_float(row.get("fighter_a_sig_strikes_landed"))
                f1_sig_attempted = to_float(row.get("fighter_a_sig_strikes_attempted"))
                f1_td_landed = to_float(row.get("fighter_a_takedowns_landed"))
                f1_td_attempted = to_float(row.get("fighter_a_takedowns_attempted"))
                f1_sub_attempts = to_float(row.get("fighter_a_submission_attempts"))
                f1_minutes = to_float(row.get("fighter_a_fight_minutes"))
                f2_sig_landed = to_float(row.get("fighter_b_sig_strikes_landed"))
                f2_sig_attempted = to_float(row.get("fighter_b_sig_strikes_attempted"))
                f2_td_landed = to_float(row.get("fighter_b_takedowns_landed"))
                f2_td_attempted = to_float(row.get("fighter_b_takedowns_attempted"))

                v1 = [
                    to_float(row.get("fighter_a_height")),
                    to_float(row.get("fighter_a_reach")),
                    to_float(row.get("fighter_a_age")),
                    (f1_sig_landed / f1_minutes) if (f1_sig_landed is not None and f1_minutes and f1_minutes > 0) else None,
                    (f1_sig_landed / f1_sig_attempted) if (f1_sig_landed is not None and f1_sig_attempted and f1_sig_attempted > 0) else None,
                    (f2_sig_landed / f1_minutes) if (f2_sig_landed is not None and f1_minutes and f1_minutes > 0) else None,
                    (1.0 - (f2_sig_landed / f2_sig_attempted)) if (f2_sig_landed is not None and f2_sig_attempted and f2_sig_attempted > 0) else None,
                    ((f1_td_landed / f1_minutes) * 15.0) if (f1_td_landed is not None and f1_minutes and f1_minutes > 0) else None,
                    (f1_td_landed / f1_td_attempted) if (f1_td_landed is not None and f1_td_attempted and f1_td_attempted > 0) else None,
                    (1.0 - (f2_td_landed / f2_td_attempted)) if (f2_td_landed is not None and f2_td_attempted and f2_td_attempted > 0) else None,
                    ((f1_sub_attempts / f1_minutes) * 15.0) if (f1_sub_attempts is not None and f1_minutes and f1_minutes > 0) else None,
                    to_float(row.get("fighter_a_weight")),
                ]
                add_record(agg, f"{wc}|{f1}", v1)
                add_class_record(class_agg, wc, v1)

            f2 = (row.get("fighter_b_name") or "").strip()
            if f2:
                f1_sig_landed = to_float(row.get("fighter_a_sig_strikes_landed"))
                f1_sig_attempted = to_float(row.get("fighter_a_sig_strikes_attempted"))
                f1_td_landed = to_float(row.get("fighter_a_takedowns_landed"))
                f1_td_attempted = to_float(row.get("fighter_a_takedowns_attempted"))
                f2_sig_landed = to_float(row.get("fighter_b_sig_strikes_landed"))
                f2_sig_attempted = to_float(row.get("fighter_b_sig_strikes_attempted"))
                f2_td_landed = to_float(row.get("fighter_b_takedowns_landed"))
                f2_td_attempted = to_float(row.get("fighter_b_takedowns_attempted"))
                f2_sub_attempts = to_float(row.get("fighter_b_submission_attempts"))
                f2_minutes = to_float(row.get("fighter_b_fight_minutes"))

                v2 = [
                    to_float(row.get("fighter_b_height")),
                    to_float(row.get("fighter_b_reach")),
                    to_float(row.get("fighter_b_age")),
                    (f2_sig_landed / f2_minutes) if (f2_sig_landed is not None and f2_minutes and f2_minutes > 0) else None,
                    (f2_sig_landed / f2_sig_attempted) if (f2_sig_landed is not None and f2_sig_attempted and f2_sig_attempted > 0) else None,
                    (f1_sig_landed / f2_minutes) if (f1_sig_landed is not None and f2_minutes and f2_minutes > 0) else None,
                    (1.0 - (f1_sig_landed / f1_sig_attempted)) if (f1_sig_landed is not None and f1_sig_attempted and f1_sig_attempted > 0) else None,
                    ((f2_td_landed / f2_minutes) * 15.0) if (f2_td_landed is not None and f2_minutes and f2_minutes > 0) else None,
                    (f2_td_landed / f2_td_attempted) if (f2_td_landed is not None and f2_td_attempted and f2_td_attempted > 0) else None,
                    (1.0 - (f1_td_landed / f1_td_attempted)) if (f1_td_landed is not None and f1_td_attempted and f1_td_attempted > 0) else None,
                    ((f2_sub_attempts / f2_minutes) * 15.0) if (f2_sub_attempts is not None and f2_minutes and f2_minutes > 0) else None,
                    to_float(row.get("fighter_b_weight")),
                ]
                add_record(agg, f"{wc}|{f2}", v2)
                add_class_record(class_agg, wc, v2)

    out = {}
    for key in sorted(agg.keys()):
        wc = key.split("|", 1)[0]
        stats = {}
        for field in FIELDS:
            total, count = agg[key][field]
            class_total, class_count = class_agg[wc][field]
            class_mean = (class_total / class_count) if class_count else 0.0
            if count:
                shrunk = (total + PRIOR_WEIGHT * class_mean) / (count + PRIOR_WEIGHT)
                stats[field] = round(shrunk, 6)
            else:
                stats[field] = round(class_mean, 6)
        out[key] = stats

    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {len(out)} fighter entries to {OUT}")


if __name__ == "__main__":
    main()
