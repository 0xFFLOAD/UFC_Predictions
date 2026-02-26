# Tesla Stock Predictions

Multi-language workspace centered on **TSLA stock forecasting**.

## Structure

- `typescript/` — TypeScript starter
- `javascript/` — JavaScript starter
- `css/` — CSS starter stylesheet
- `python/` — Python starter script
- `c99/` — C99 starter example
- `data/` — Tesla OHLCV dataset + feature utilities
- `model/` — C99 forecasting model (`tsla_nn.c`)
- `odds/` — Forecast table artifacts (`odds.txt`)

## Quick Start

```bash
cd model
make train
make predict
```

Generate forecast table:

```bash
make odds-table
```

## Data

Primary dataset:

- `data/TSLA_Stock_Dataset_2012_2026.csv` — Daily TSLA OHLCV data

Preview and feature snapshot:

```bash
python3 data/preview_data.py
python3 data/build_tesla_feature_snapshot.py
```

See [model/README.md](model/README.md) and [data/README.md](data/README.md) for details.
