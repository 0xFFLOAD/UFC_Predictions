# Tesla Stock Forecast Model Makefile

CC = cc
CFLAGS = -std=c99 -Wall -Wextra -pedantic -O3 -march=native
LDFLAGS = -lm -lpthread
TARGET = tsla_nn
MODEL_BIN = tsla_model.bin

# Source files
SRC = tsla_nn.c
OBJ = $(SRC:.c=.o)

.PHONY: all clean train predict eval test odds-table

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Train the model from scratch
train: $(TARGET)
	@echo "Training Tesla forecast model..."
	./$(TARGET) --train

# Load existing model and enter prediction mode
predict: $(TARGET)
	@echo "Generating Tesla next-period forecast..."
	./$(TARGET) --load --predict

# Evaluate on chronological holdout split
eval: $(TARGET)
	@echo "Running historical evaluation..."
	/usr/bin/python3 evaluate.py

# Generate odds table file from matchup inputs
odds-table:
	@echo "Generating forecast table..."
	@set -e; \
	NORM_ARGS=""; \
	EXPECT_OUT_PATH=0; \
	for token in $(ARGS); do \
		if [ $$EXPECT_OUT_PATH -eq 1 ]; then \
			case "$$token" in \
				/*|./*|../*) out_path="$$token" ;; \
				*) out_path="../$$token" ;; \
			esac; \
			NORM_ARGS="$$NORM_ARGS --out $$out_path"; \
			EXPECT_OUT_PATH=0; \
			continue; \
		fi; \
		case "$$token" in \
			--out) \
				EXPECT_OUT_PATH=1 ;; \
			--out=*) \
				out_raw=$${token#--out=}; \
				case "$$out_raw" in \
					/*|./*|../*) out_path="$$out_raw" ;; \
					*) out_path="../$$out_raw" ;; \
				esac; \
				NORM_ARGS="$$NORM_ARGS --out=$$out_path" ;; \
			*) \
				NORM_ARGS="$$NORM_ARGS $$token" ;; \
		esac; \
	done; \
	if [ $$EXPECT_OUT_PATH -eq 1 ]; then \
		echo "Error: --out flag provided without a path"; \
		exit 2; \
	fi; \
	/usr/bin/python3 ../odds/generate_odds_table.py $$NORM_ARGS

# Train and then predict
train-predict: $(TARGET)
	@echo "Training and generating immediate forecast..."
	./$(TARGET) --train && ./$(TARGET) --load --predict

# Quick test with limited data
test: $(TARGET)
	@echo "Testing forecast binary..."
	./$(TARGET) --predict

clean:
	rm -f $(TARGET) $(OBJ) $(MODEL_BIN)

rebuild: clean all

# Benchmark training performance
benchmark: $(TARGET)
	@echo "Benchmarking training..."
	time ./$(TARGET)

# Show model info
info:
	@echo "=== Tesla Forecast Model ==="
	@echo "Input: TSLA OHLCV monthly history"
	@echo "Output: next-period close forecast + direction confidence"
	@echo ""
	@ls -lh $(MODEL_BIN) 2>/dev/null || echo "No trained model found (run 'make train')"

help:
	@echo "Tesla Stock Forecast Model"
	@echo ""
	@echo "Usage:"
	@echo "  make              Build the model"
	@echo "  make train        Train baseline TSLA return model"
	@echo "  make predict      Forecast next period from trained model"
	@echo "  make eval         Evaluate directional hit rate"
	@echo "  make odds-table   Generate forecast table at odds/odds.txt"
	@echo "                    Optional: make odds-table ARGS='--out odds/odds.txt'"
	@echo "                    Non-interactive example: /usr/bin/python3 ../odds/generate_odds_table.py --date 2026-03-08T00:00"
	@echo "  make train-predict Train then predict"
	@echo "  make test         Quick test run"
	@echo "  make clean        Remove built files"
	@echo "  make benchmark    Benchmark training performance"
	@echo "  make info         Show model information"
