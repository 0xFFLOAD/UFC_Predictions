# UFC Winner Prediction Neural Network
A high-performance neural network written in pure C99 that predicts UFC fight outcomes based on fighter statistics.

## Architecture
Input Layer:    25 features (fighter stat deltas and features)
Hidden Layer 1: 64 neurons (tanh activation)
Hidden Layer 2: 32 neurons (tanh activation)
Output Layer:   1 neuron (sigmoid) - P(fighter1 wins)

## Features
The model uses **25 carefully engineered features** representing differences between fighters:

### Basic Physical Deltas
1. **Height Delta** - Height difference (cm)
2. **Reach Delta** - Reach difference (cm)
3. a. **Age Delta** - Age difference (months)
4. b. **Age** - Age (months)
5. a. **Weight Delta** - Weight difference (kg)
6. b. **Weight** - Weight (kg)

### Striking Statistics Deltas
7. **Sig Strikes PM Delta** - Significant strikes landed per minute
8. **Sig Strike Accuracy Delta** - Strike accuracy percentage
9. **Sig Strike Absorbed Delta** - Strikes absorbed per minute
10. **Sig Strike Defense Delta** - Strike defense percentage

### Grappling Statistics Deltas
11. **Takedown Avg Delta** - Takedowns averaged per 15 minutes
12. **Takedown Accuracy Delta** - Takedown success rate
13. **Takedown Defense Delta** - Takedown defense rate
14. **Submission Avg Delta** - Submission attempts per 15 minutes

### Derived Composite Features
15. **Striking Advantage** - Net striking effectiveness (output - absorbed)
16. **Grappling Score** - Composite grappling effectiveness

### Miscellaneous
17. a. **Unprepared for Fight** - Short-notice fight (binary)
18. b. **Win/Loss Unprepared for Fight** - Short-notice fight (binary)
19. **Total Injuries in Career** - Previous injuries
20. **Total Consecutive Wins** - Win streak
21. a. **Total Special Win count** - Reason for win (opp )
22. b. **Total Special Loss count** - Reason for loss (self sick, self injured)
23. **Total count KO taken** - Total count of KO taken
24. **Total count KO given** - Total count of KO given
25. **Prev Wins Against Opp** - Previous fights against opponent

## Dataset


## Building
make              # Build the model
make train        # Train from scratch
make predict      # Interactive prediction mode
make test         # Quick test
make clean        # Clean build artifacts

## Usage

### Training the Model

### Interactive Prediction

## Model Performance

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

## Technical Notes

### Feature Engineering Rationale

### Normalization

### Activation Functions
- **Hidden layers**: `tanh` (symmetric, works well for deltas)
- **Output layer**: `sigmoid` (probability output in [0, 1])

### Loss Function
**Binary cross-entropy:**
  L = -y*log(ŷ) - (1-y)*log(1-ŷ)

## Limitations

## Future Improvements

## References

## License

**Disclaimer**: This model is for educational and research purposes only. Do not use for sports betting. MMA outcomes are highly unpredictable.