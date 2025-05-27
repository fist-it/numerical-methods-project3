interpolate: format
	@python3 interpolation.py

run:
	@python3 main.py

test:
	python3 test.py

format:
	@ruff format .
