RUNS	:= runs
SUBMIT	:= submit.csv

.PHONY: clean

clean:
	rm -rf $(RUNS) $(SUBMIT)