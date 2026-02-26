.PHONY: odds-table help

# Forward odds table generation to model Makefile
odds-table:
	$(MAKE) -C model odds-table ARGS="$(ARGS)"

help:
	@echo "tesla_stock_predictions root shortcuts"
	@echo ""
	@echo "  make odds-table                 Generate forecast table at odds/odds.txt"
	@echo "  make odds-table ARGS='--out odds/odds.txt'"
