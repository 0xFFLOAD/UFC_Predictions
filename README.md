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
- `data/` — **UFC datasets and data utilities**
- `model/` — **Neural network for UFC winner prediction** (C99, 14 features)
- `odds/` — Processed betting-odds artifacts and reports

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

Primary UFC datasets in this workspace:

- `ufc_complete_dataset.csv` — Core UFC fight dataset used by the model
- `ufc_complete_dataset.full_backup.csv` — Backup copy of the complete dataset
- `ufc_fights_full_with_odds.csv` — UFC fights with integrated betting odds
- `fighter_stats_dict.json` — Fighter stat dictionary built from source data

**Preview the data:**
```bash
cd data
python3 preview_data.py
```

## Odds Reports

- Processed odds report output is stored at `odds/processed_odds_report.txt`.


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
