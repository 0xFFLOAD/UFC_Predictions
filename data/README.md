# UFC Data Sources

This directory currently contains a class-filtered UFC dataset for only:

- `Welterweight`
- `Featherweight`
- `Bantamweight`
- `Lightweight`

## Dataset

### `ufc_complete_dataset.csv`
**Source:** [jansen88/ufc-data](https://github.com/jansen88/ufc-data)  
**Original Coverage:** 1994 to 2023 UFC events  
**Current Workspace Coverage:** Filtered to 4 weight classes only  
**Rows in this workspace:** 3,836 (post-filter)  
**License:** Public (open source repository)

## Removed Files

The following files were removed from this workspace because they were empty after class filtering:

- `ufc_fight_data.csv`
- `ufc_fight_data_raw.csv`

## Data Dictionary (ufc_complete_dataset.csv)

| Field | Example | Description | Source |
|-------|---------|-------------|--------|
| `event_date` | 2023-09-16 | Date of UFC event | UFC Stats |
| `event_name` | UFC Fight Night: Grasso vs. Shevchenko 2 | Event name | UFC Stats |
| `weight_class` | Welterweight | Weight class | UFC Stats |
| `fighter1`, `fighter2` | Alexa Grasso, Valentina Shevchenko | Fighter names | UFC Stats |
| `favourite`, `underdog` | Valentina Shevchenko, Alexa Grasso | Betting favorites | betmma.tips |
| `favourite_odds`, `underdog_odds` | 1.67, 2.88 | Decimal odds | betmma.tips |
| `betting_outcome` | favourite/underdog | Betting result | Derived |
| `outcome` | fighter1/fighter2/Draw | Match winner | UFC Stats |
| `method` | S-DEC, U-DEC, KO/TKO | Victory method | UFC Stats |
| `round` | 5 | Round of victory | UFC Stats |
| `fighter1_height` | 165.1 | Height (cm) | UFC Stats |
| `fighter1_reach` | 167.64 | Reach (cm) | UFC Stats |
| `fighter1_stance` | Orthodox/Southpaw | Fighting stance | UFC Stats |
| `fighter1_sig_strikes_landed_pm` | 4.67 | Sig strikes/min | UFC Stats |
| `fighter1_sig_strikes_accuracy` | 0.43 | Strike accuracy (%) | UFC Stats |
| `fighter1_takedown_avg_per15m` | 0.41 | Takedowns per 15min | UFC Stats |
| `fighter1_takedown_accuracy` | 0.45 | Takedown accuracy (%) | UFC Stats |
| `fighter1_takedown_defence` | 0.59 | Takedown defense (%) | UFC Stats |

(Same fields available for `fighter2_*`)

## Usage Examples

### Python
```python
import pandas as pd

# Load comprehensive dataset
df = pd.read_csv('data/ufc_complete_dataset.csv')
print(f"Total fights: {len(df)}")
print(f"Date range: {df['event_date'].min()} to {df['event_date'].max()}")

# Filter by allowed classes
allowed = {"Welterweight", "Featherweight", "Bantamweight", "Lightweight"}
df = df[df['weight_class'].isin(allowed)]

# Analyze betting accuracy
betting_data = df.dropna(subset=['betting_outcome'])
fav_win_rate = (betting_data['betting_outcome'] == 'favourite').mean()
print(f"Favorite win rate: {fav_win_rate:.1%}")
```

### JavaScript
```javascript
const fs = require('fs');
const parse = require('csv-parse/sync');

const data = fs.readFileSync('data/ufc_complete_dataset.csv');
const records = parse.parse(data, { columns: true });

console.log(`Total fights: ${records.length}`);
```

### TypeScript
```typescript
import * as fs from 'fs';
import { parse } from 'csv-parse/sync';

interface UFCFight {
  event_date: string;
  fighter1: string;
  fighter2: string;
  outcome: string;
  method: string;
}

const data = fs.readFileSync('data/ufc_complete_dataset.csv');
const fights: UFCFight[] = parse(data, { columns: true });
```

## Data Quality Notes

- Dataset is intentionally restricted to 4 classes for model specialization.
- Betting odds are only present for parts of the timeline.
- Fighter names may have minor variations across sources.
- Data is historical and scraped from public UFC-related sources.

## Additional Resources

- **Official UFC Stats:** https://ufcstats.com/
- **UFC API Projects:** 
  - https://github.com/jgcmarins/graphql-ufc-api
  - https://github.com/sungwoncho/node-ufc-api
- **Data Analysis Examples:**
  - https://github.com/komaksym/UFC-DataLab
  - https://github.com/petermartens98/UFC-Rankings-EDA-1993-to-2022

## License

All datasets are sourced from public repositories and scraped from publicly accessible sources. Please review individual repository licenses if using for commercial purposes.
