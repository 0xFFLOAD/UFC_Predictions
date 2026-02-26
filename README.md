# UFC Multi-Language Starter

Starter workspace for:
- TypeScript
- JavaScript
- CSS
- Python
- C99

## Structure

- `typescript/` — TypeScript app with `tsconfig.json`
- `javascript/` — JavaScript starter
- `css/` — CSS starter stylesheet
- `python/` — Python starter script
- `c99/` — C99 starter + `Makefile`
- `data/` — **Real UFC datasets** (30 years of fights, fighter stats, betting odds)
- `model/` — **Neural network for UFC winner prediction** (C99, 14 features)

## Quick Start

### TypeScript
```bash
cd typescript
npm install
npm run build
npm run start
```

### JavaScript
```bash
cd javascript
node index.js
```

### CSS
Use `css/styles.css` in any HTML page.

### Python
```bash
cd python
python3 main.py
```

### C99
```bash
cd c99
make
make run
```

## Data

Real UFC datasets with **32,000+ rows** across 3 files:

- `ufc_complete_dataset.csv` — 7,340 fights from 1994-2023 with fighter stats & betting odds
- `ufc_fight_data.csv` — 17,052 rows of detailed fight statistics (updated weekly)
- `ufc_fight_data_raw.csv` — 8,526 rows of raw UFC stats data

**Preview the data:**
```bash
cd data
python3 preview_data.py
```


## Neural Network Model

High-performance **C99 neural network** that predicts UFC fight winners:

**Architecture:** 14 features → 64 → 32 → 1 (sigmoid output)

**Build and train:**
```bash
cd model
make train        # Train on 7,340 UFC fights
make predict      # Interactive prediction mode
```

**Features used:**
- Height, reach, age, weight deltas
- Striking statistics (accuracy, output, defense)
- Takedown and submission statistics
- Derived features (striking advantage, grappling score)

**Expected accuracy:** 70-75% on historical fight outcomes

See [model/README.md](model/README.md) for architecture details and usage.
See [data/README.md](data/README.md) for detailed documentation on fields, sources, and usage examples.
