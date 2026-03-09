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
1.  **Height Delta** - Height difference (cm)
2.  **Reach Delta** - Reach difference (cm)
3.  **Age Delta** - Age difference (months)
4.  **Age** - Age (months)
5.  **Weight Delta** - Weight difference (kg)
6.  **Weight** - Weight (kg)

### Striking Statistics Deltas
7.  **Sig Strikes PM Delta** - Significant strikes landed per minute
8.  **Sig Strike Accuracy Delta** - Strike accuracy percentage
9.  **Sig Strike Absorbed Delta** - Strikes absorbed per minute
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
17. **Unprepared for Fight** - Short-notice fight (binary)
18. **Win/Loss Unprepared for Fight** - Short-notice fight (binary)
19. **Total Injuries in Career** - Previous injuries
20. **Total Consecutive Wins** - Win streak
21. **Total Special Win count** - Reason for win (opp )
22. **Total Special Loss count** - Reason for loss (self sick, self injured)
23. **Total count KO taken** - Total count of KO taken
24. **Total count KO given** - Total count of KO given
25. **Prev Wins Against Opp** - Previous fights against opponent

## Dataset

## Data Extraction Scripts
A companion `extract` directory at the project root now organizes
each feature into its own Python package. For example, the height delta
extractor lives in `extract/height_delta/height_delta.py` and is
exported through the package's `__init__.py`.

Each package exposes a single `extract()` function that reads the large
set (`ufc_dataset/large_set/large_dataset.csv`) and returns the columns
or derived values for that feature.  If the underlying dataset lacks the
requested data, the function prints a warning and returns an empty
DataFrame.

These packages can be imported directly for use in pipelines or run as
scripts for quick previews.

Example usage::

    from extract.height_delta import extract as extract_height
    df = extract_height()

or (using the package alias)::

    import extract.height_delta
    df = extract.height_delta.extract()

When invoked as a script, each extractor writes the complete DataFrame
it returns to a TSV file named `<feature>.tsv` inside its own package
directory. Every output file now includes the fighter identifiers
(`r_fighter`, `b_fighter`) along with the `winner` column so that all
data stays tied to the participants and outcome; features themselves
are appended or computed as additional columns. This avoids
orphaned/random rows when working with individual feature files.

For the age extractor specifically, the output is further split by
weight class: running `python extract/age/age.py` creates the main
`age.tsv` plus additional files like `age_Bantamweight.tsv`,
`age_Heavyweight.tsv`, etc.  The separation reflects the fact that
fighters from different classes never meet, so models can be trained
on each class independently.  Simply point the training script at the
appropriate class-specific TSV when you want to focus on one category.

The new age‑delta extractor behaves the same way: `python
extract/age_delta/age_delta.py` writes both `age_delta.tsv` and a
collection of `age_delta_<class>.tsv` files so you can test the
influence of pure age differences within or across weight divisions.

The extractors rely on **pandas**. If you attempt to run one without
pandas installed you'll see an error prompting installation. On macOS
the system Python is managed by Homebrew and will refuse a normal
`pip install`; instead use one of the following approaches:

> **Note:** the training helpers (`train.py`, `train_by_class.py`,
> `eval_by_class.py`) also require PyTorch; install it in the same
> environment as pandas, either via `pip install torch` or the appropriate
> `conda` command.

* Activate the project's Conda environment (already used by this
  workspace). For example:

```bash
/Users/sam/miniforge3/bin/conda run -p /Users/sam/miniforge3 \
    --no-capture-output python extract/height_delta/height_delta.py
```

  The `install_python_packages` tool has already added pandas to this
environment, so scripts will run out‑of‑the‑box.

* Create and activate a virtualenv in the repo, then `pip install
  pandas` inside it:

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas
python extract/height_delta/height_delta.py
```

* Alternatively, install pandas for your user only with `pip install
  --user pandas` or use `pipx`.

Example (assuming pandas is available):

```bash
python3 extract/height_delta/height_delta.py
# creates extract/height_delta/height_delta.tsv
```

### Model Implementation & Training
A basic PyTorch implementation of the UFC prediction network now
lives in `model/neural_network.py` with a convenience training script
at `model/train.py`.

#### Requirements
Install PyTorch in the same environment used for pandas. For example:

```bash
# use pip or conda depending on your setup
pip install torch           # CPU-only
# or
conda install pytorch -c pytorch
```

#### Training via script
Supply one or more TSV feature files and list the columns to use as
inputs. When multiple files are given the script will **join them
horizontally** on the fighter identifiers (`r_fighter`, `b_fighter`
(and `weight_class` if present) ) rather than stacking rows.  This
makes it easy to experiment with combinations of features without
manually merging datasets.

Examples:

* train solely on age-differences:

```bash
python model/train.py \
    --data extract/age_delta/age_delta.tsv \
    --features age_diff \
    --epochs 50 --lr 0.001
```

* train on absolute age and the pre‑computed age delta together:

```bash
python model/train.py \
    --data extract/age/age.tsv extract/age_delta/age_delta.tsv \
    --features r_age b_age age_diff \
    --epochs 50 --lr 0.001
```

The training routine automatically converts the `winner` column into a
binary label (`Red`=1, `Blue`=0) and shuffles data during training.  A
few additional options help when you want to flip the prediction target
or generate both models in one go:

* `--invert` treats `Blue` victories as the positive class (i.e. you
   are predicting the loss of the red fighter).  This simply inverts the
   label during training and evaluation.
* `--double` runs the training twice: first predicting wins, then
   predicting losses.  Two checkpoint files are written with suffixes
   `_win.pt` and `_loss.pt` (or a custom `--save` prefix if you provide
   one).
* `--save <prefix>` specifies a path prefix where the model(s) should be
   saved.  When using `--double` the `_win.pt`/`_loss.pt` suffixes are
   appended automatically; otherwise `_win.pt` is the default.

These flags work alongside `--auto-lr`, `--search`, and
`--per-class`.

#### Hyperparameter grid search
To avoid blind guessing of lr/batch/epoch values you can ask the script
to sweep a small grid.  Supply comma-separated lists for each parameter
and include `--search` on the command line.  For example:

```bash
python model/train.py --search \
    --data extract/age/age.tsv \
    --features r_age b_age \
    --lr-values 1e-4,5e-4,1e-3 \
    --batch-values 16,32,64 \
    --epoch-values 20,50
```

The script will train one model for each combination and report the
best configuration (lowest loss) at the end.

#### Automatic learning‑rate finder
If you only care about picking a sensible learning rate you can let the
script estimate it for you.  Use the `--auto-lr` flag and it will run a
short, internal sweep before the main training pass:

```bash
python model/train.py --auto-lr \
    --data extract/age/age.tsv --features r_age b_age \
    --epochs 20 --batch 32
```

It will print the suggested lr and then continue training with that
value.  This removes the need to hand‑tune a value while still keeping
control over the number of epochs and batch size.
#### Per‑weight‑class training
Since fighters only face opponents in the same weight class, the
script can automatically split the dataset by `weight_class` and
train a separate network on each segment.  Add `--per-class` and the
remaining arguments (including `--auto-lr` or `--search`) are applied
individually to each class.  Example:

```bash
python model/train.py --per-class --data extract/age/age.tsv \
    --features r_age b_age --epochs 50 --batch 32
```

If you supply multiple data files the same merge rules described above
are used before the per-class split.  For instance, to train each
weight class on both absolute age and age‑difference:

```bash
python model/train.py --per-class \
    --data extract/age/age.tsv extract/age_delta/age_delta.tsv \
    --features r_age b_age age_diff \
    --epochs 50 --batch 32
```

This will print a loss for each weight class and a summary at the end.

The helper script `extract/age/train_by_class.py` offers the same
overlapping-feature merging and, in addition, understands `--invert` and
`--double` so you can produce loss models (or both win/loss) for every
class in one shot.  The evaluation helper similarly supports inverted
labels.

A companion evaluation helper exists at `extract/age/eval_by_class.py`.
Run it from that directory to compute accuracy/loss against the same
per-class TSVs (or pass `--data` to merge multiple feature tables
first).  You can also use `--invert` to evaluate _loss_ models, and the
script will automatically detect files whose names end in `_loss.pt`.
It is convenient for quickly comparing how different feature sets behave
on each division.
#### Programmatic use
You can also import the classes directly:

```python
from model.neural_network import UFCPredictor, train_model
import pandas as pd

df = pd.read_csv('extract/age/age.tsv', sep='\t')
features = ['r_age', 'b_age']
model = UFCPredictor(input_dim=len(features))
train_model(model, df, features)
```




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