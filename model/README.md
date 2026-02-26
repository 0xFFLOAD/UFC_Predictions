# Tesla Forecast Model (C99)

This folder contains a C99 baseline forecaster for TSLA stock using historical OHLCV data.

## What it does

- Loads `data/TSLA.csv` (fallback: `../data/TSLA.csv`)
- Computes average monthly return and return volatility
- Saves model parameters to `tsla_model.bin`
- Predicts next-period close and bullish probability

## Build and run

```bash
make
make train
make predict
make eval
```

## Forecast output

`make predict` prints:

- Forecast date
- Last close
- Expected return (%)
- Predicted close
- Bullish probability (%)

## Files

- `tsla_nn.c` — forecasting binary source
- `Makefile` — build and run commands
- `evaluate.py` — directional hit-rate evaluation
- `demo.sh` / `quickstart.sh` — convenience scripts

## Notes

- This is a lightweight baseline, not financial advice.
- Extend with additional indicators (RSI, MACD, market regime, macro factors) for stronger modeling.
