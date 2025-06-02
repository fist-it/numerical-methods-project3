latex-compile:
	pdflatex -output-directory="./sprawozdanie" ./sprawozdanie.tex

run:
	@python3 main.py

test_data:
	@python3 data.py

interpolate: format
	@python3 interpolation.py

test:
	python3 test.py

format:
	@ruff format .
