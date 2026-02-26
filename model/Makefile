# UFC Winner Prediction Neural Network Makefile

CC = cc
CFLAGS = -std=c99 -Wall -Wextra -pedantic -O3 -march=native
LDFLAGS = -lm -lpthread
TARGET = ufc_nn

# Source files
SRC = ufc_nn.c
OBJ = $(SRC:.c=.o)

.PHONY: all clean train predict matchup eval test odds-table

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Train the model from scratch
train: $(TARGET)
	@echo "Training UFC winner prediction models by weight class..."
	./$(TARGET)

# Train a single global model
train-global: $(TARGET)
	@echo "Training single global UFC model..."
	./$(TARGET) --global

# Load existing model and enter prediction mode
predict: $(TARGET)
	@echo "Starting interactive prediction mode..."
	./$(TARGET) --load --predict

# Analyze a specific fighter-vs-fighter matchup
matchup: $(TARGET)
	@echo "Starting matchup analysis mode..."
	./$(TARGET) --load --matchup

# Evaluate on chronological holdout split
eval: $(TARGET)
	@echo "Running model evaluation on chronological holdout..."
	./$(TARGET) --global --load --eval

# Generate odds table file from matchup inputs
odds-table:
	@echo "Generating odds table..."
	/usr/bin/python3 ../odds/generate_odds_table.py $(ARGS)

# Train and then predict
train-predict: $(TARGET)
	@echo "Training and entering prediction mode..."
	./$(TARGET) && ./$(TARGET) --load --predict

# Quick test with limited data
test: $(TARGET)
	@echo "Testing model architecture..."
	./$(TARGET) --load 2>/dev/null || ./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJ) ufc_model.bin

rebuild: clean all

# Benchmark training performance
benchmark: $(TARGET)
	@echo "Benchmarking training..."
	time ./$(TARGET)

# Show model info
info:
	@echo "=== UFC Prediction Neural Network ==="
	@echo "Architecture: 20 -> 64 -> 32 -> 1"
	@echo "Input features: Fighter stat deltas"
	@echo "Output : P(Fighter 1 wins)"
	@echo "Dataset: 30 years of UFC history"
	@echo ""
	@ls -lh ufc_model.bin 2>/dev/null || echo "No trained model found (run 'make train')"

help:
	@echo "UFC Winner Prediction Neural Network"
	@echo ""
	@echo "Usage:"
	@echo "  make              Build the model"
	@echo "  make train        Train one model per weight class"
	@echo "  make train-global Train one global model"
	@echo "  make predict      Load model and predict fights (defaults to future mode)"
	@echo "  ./ufc_nn --load --predict-future      Predict hypothetical future fights"
	@echo "  ./ufc_nn --load --predict-historical  Predict with direct head-to-head enabled"
	@echo "  make matchup      Compare model odds vs actual A vs B outcomes"
	@echo "  make eval         Evaluate global model on holdout split"
	@echo "  make odds-table   Generate odds/odds.txt (interactive by default)"
	@echo "                    Optional: make odds-table ARGS='--out ../odds/odds.txt'"
	@echo "                    Non-interactive example: /usr/bin/python3 ../odds/generate_odds_table.py --pair 'Lightweight|Max Holloway|Charles Oliveira'"
	@echo "  make train-predict Train then predict"
	@echo "  make test         Quick test run"
	@echo "  make clean        Remove built files"
	@echo "  make benchmark    Benchmark training performance"
	@echo "  make info         Show model information"
