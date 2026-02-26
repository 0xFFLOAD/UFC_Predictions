# Tesla Data Sources

This directory contains Tesla stock data and lightweight preprocessing utilities.

## Dataset

### `TSLA_Stock_Dataset_2012_2026.csv`

Required columns:

- `Date` (`YYYY-MM-DD`)
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

The model treats this file as a chronological daily OHLCV time series.

## Utilities

- `preview_data.py` — quick profile of date range, return statistics, and volatility.
- `build_tesla_feature_snapshot.py` — computes rolling indicators and exports JSON snapshot.

## Usage

```bash
python3 data/preview_data.py
python3 data/build_tesla_feature_snapshot.py
```

## Notes

- Keep rows sorted by `Date` ascending.
- Remove commas from `Volume` values if present.
- Missing days (weekends/holidays) are fine.
