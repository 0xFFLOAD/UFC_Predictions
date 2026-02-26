.PHONY: odds-table help

# Forward odds table generation to model Makefile
odds-table:
	$(MAKE) -C model odds-table ARGS="$(ARGS)"

help:
	@echo "UFC_Predictions root shortcuts"
	@echo ""
	@echo "  make odds-table                 Generate odds/odds.txt (interactive)"
	@echo "  make odds-table ARGS='--out odds/odds.txt'"
