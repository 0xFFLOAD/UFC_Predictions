# UFC Winner Prediction Neural Network

A high-performance neural network written in pure C99 that predicts UFC fight outcomes based on fighter statistics.

## Architecture

```
Input Layer:    14 features (fighter stat deltas)
Hidden Layer 1: 64 neurons (tanh activation)
Hidden Layer 2: 32 neurons (tanh activation)
Output Layer:   1 neuron (sigmoid) - P(fighter1 wins)
```

## Features

The model uses **14 carefully engineered features** representing differences between fighters:

### Basic Physical Deltas
1. **Height Delta** - Height difference (cm)
2. **Reach Delta** - Reach difference (cm)
3. **Age Delta** - Age difference (years)
4. **Weight Delta** - Weight difference (kg)

### Striking Statistics Deltas
5. **Sig Strikes PM Delta** - Significant strikes landed per minute
6. **Sig Strike Accuracy Delta** - Strike accuracy percentage
7. **Sig Strike Absorbed Delta** - Strikes absorbed per minute
8. **Sig Strike Defense Delta** - Strike defense percentage

### Grappling Statistics Deltas
9. **Takedown Avg Delta** - Takedowns averaged per 15 minutes
10. **Takedown Accuracy Delta** - Takedown success rate
11. **Takedown Defense Delta** - Takedown defense rate
12. **Submission Avg Delta** - Submission attempts per 15 minutes

### Derived Composite Features
13. **Striking Advantage** - Net striking effectiveness (output - absorbed)
14. **Grappling Score** - Composite grappling effectiveness

## Dataset

Trained on **30 years of UFC history** (1994-2023):
- **7,340 fights** from `ufc_complete_dataset.csv`
- 2,434 unique fighters
- Comprehensive fighter statistics
- Real fight outcomes

## Building

```bash
make              # Build the model
make train        # Train from scratch
make predict      # Interactive prediction mode
make test         # Quick test
make clean        # Clean build artifacts
```

## Usage

### Training the Model

```bash
make train
```

This will:
1. Load all UFC fight data from `../data/ufc_complete_dataset.csv`
2. Compute feature normalization statistics
3. Train the neural network for up to 500 epochs
4. Save the trained model to `ufc_model.bin`

Example output:
```
=== Training UFC Winner Prediction Model ===
Samples: 7340
Architecture: 14 -> 64 -> 32 -> 1

Training for up to 500 epochs...
Epoch   1/500  Loss: 0.693147  Accuracy: 50.27%  LR: 0.010000
Epoch  10/500  Loss: 0.621543  Accuracy: 65.41%  LR: 0.009512
Epoch  20/500  Loss: 0.587921  Accuracy: 68.35%  LR: 0.009048
...
Final accuracy: 72.18%
```

### Interactive Prediction

```bash
make predict
```

Enter fighter statistics to get win probability predictions:

```
=== UFC Fight Predictor ===
Enter fighter statistics to predict winner probability

Fighter 1 stats (height reach age sig_str_pm takedown_avg): 180 185 28 4.5 2.1
Fighter 2 stats (height reach age sig_str_pm takedown_avg): 175 178 32 3.8 1.5

--- Prediction ---
Fighter 1 win probability: 67.32%
Fighter 2 win probability: 32.68%
Predicted winner: Fighter 1
```

### Evaluation

```bash
python3 evaluate.py
```

Evaluates the trained model on test data and shows sample predictions.

## Model Performance

Expected performance metrics:
- **Training Accuracy**: 70-75%
- **Generalization**: MMA is highly unpredictable; the model captures statistical trends
- **Key Insights**: 
  - Striking advantage is the strongest predictor
  - Age delta matters (younger fighters have an edge)
  - Reach and takedown statistics contribute significantly

## Implementation Details

### Optimization Techniques
- **He initialization** for weight initialization
- **Momentum SGD** with 0.9 momentum
- **Learning rate decay** (0.95 every 20 epochs)
- **Early stopping** with patience of 50 epochs
- **Feature normalization** (z-score standardization)
- **Batch shuffling** for better convergence

### Performance
- **Multi-threaded training** for large datasets (via pthreads)
- **Optimized compilation** with `-O3 -march=native`
- **Efficient memory layout** using C structs

### Portability
- Pure C99 standard code
- POSIX threads for parallelization
- Works on macOS, Linux, and BSD systems

## File Structure

```
model/
├── ufc_nn.c           # Main neural network implementation
├── Makefile           # Build system
├── evaluate.py        # Evaluation script
├── README.md          # This file
└── ufc_model.bin      # Trained model (generated after training)
```

For processed betting-odds report artifacts, see `../odds/README.md`.

## Technical Notes

### Feature Engineering Rationale

**Why deltas?** The model predicts relative matchup outcomes. By using deltas (fighter1 - fighter2), the network learns which fighter has advantages in each dimension.

**Why derived features?** Combat sports involve complex interactions:
- **Striking Advantage** captures net offensive effectiveness
- **Grappling Score** combines takedown ability with submission threat

### Normalization

All features are z-score normalized:
```c
normalized_feature = (feature - mean) / std_dev
```

This ensures all features contribute equally to learning regardless of magnitude.

### Activation Functions
- **Hidden layers**: `tanh` (symmetric, works well for deltas)
- **Output layer**: `sigmoid` (probability output in [0, 1])

### Loss Function
Binary cross-entropy:
```
L = -y*log(ŷ) - (1-y)*log(1-ŷ)
```

## Limitations

1. **Dataset imbalance**: More data for certain weight classes
2. **Career trajectory**: Stats don't capture career arc (rising vs declining)
3. **Intangibles**: Fight IQ, heart, chin, gameplan not captured
4. **Sample size**: Some fighters have limited historical data
5. **MMA unpredictability**: Upsets are common (knockouts, submissions)

## Future Improvements

- [ ] Add fight streak features (win/loss streaks)
- [ ] Include finish rate (KO/submission percentage)
- [ ] Weight class one-hot encoding
- [ ] Fighter style classification (striker/wrestler/grappler)
- [ ] Betting odds integration as feature
- [ ] Ensemble methods (multiple models voting)
- [ ] Deep learning variant (more layers)
- [ ] Recurrent network for career trajectory

## References

- Dataset: [andrew-couch/UFC_Data](https://github.com/andrew-couch/UFC_Data)
- Dataset: [jansen88/ufc-data](https://github.com/jansen88/ufc-data)
- UFC Stats: [http://ufcstats.com/](http://ufcstats.com/)

## License

Public domain. Use freely for research, education, or entertainment.

**Disclaimer**: This model is for educational and research purposes only. Do not use for sports betting. MMA outcomes are highly unpredictable.
