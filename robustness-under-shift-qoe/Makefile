.PHONY: install lint test run

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -q

run:
	python scripts/run_experiment.py --config configs/default.yaml --method erm
