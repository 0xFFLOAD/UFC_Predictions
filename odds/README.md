# Odds Artifacts

This directory stores processed betting-odds outputs used by the UFC prediction workflow.

## Files

- `processed_odds_report.txt` — Generated report with processed odds summary.
- `odds.txt` — Generated matchup odds table.
- `generate_odds_table.py` — Interactive table generator for fighter matchups.

## Generate `odds.txt`

From repository root:

```bash
python3 odds/generate_odds_table.py
```

Then enter rows interactively:

1. Weight class
2. Fighter A name
3. Fighter B name

Type `q` for weight class when done. The script writes output to `odds/odds.txt`.

By default, rows are only written when at least one fighter has odds `>= 70%`.
Rows with missing fighter stats/odds (`N/A`) are not written to the output table.
Rows include `Date` and `Time` when the matchup is found in the built-in schedule mapping.

Optional non-interactive mode:

```bash
python3 odds/generate_odds_table.py \
	--pair "Lightweight|Max Holloway|Charles Oliveira" \
	--pair "Bantamweight|Marlon Vera|David Martinez"
```

Adjust threshold if needed:

```bash
python3 odds/generate_odds_table.py --min-odds 75
```

## Notes

- This directory is intended for derived artifacts, not raw source datasets.
- Source UFC datasets remain in `data/`.
