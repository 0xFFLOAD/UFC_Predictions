# Forecast Artifacts

This directory stores generated TSLA forecast tables.

## Files

- `odds.txt` — forecast output table (date/time, projected close, return, confidence)
- `generate_odds_table.py` — forecast table generator

## Generate `odds.txt`

Interactive:

```bash
python3 odds/generate_odds_table.py
```

Non-interactive:

```bash
python3 odds/generate_odds_table.py \
  --date 2026-03-08T00:00 \
  --date 2026-03-21T19:00
```

Rows are appended to existing output by default.

## Filters

- `--min-confidence` controls minimum directional confidence to keep a row.
- Default: `55`.
