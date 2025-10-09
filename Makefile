.PHONY: install test lint format clean docs help

help:
	@echo "Available commands:"
	@echo "  make install    - Install package and dependencies"
	@echo "  make test       - Run tests with coverage"
	@echo "  make lint       - Run linters (flake8, mypy)"
	@echo "  make format     - Format code (black, isort)"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make docs       - Build documentation"
	@echo "  make publish    - Publish to PyPI"

install:
	pip install -e ".[dev]"

test:
	pytest --cov=plackett_luce --cov-report=html --cov-report=term-missing

test-fast:
	pytest -x -v

lint:
	flake8 plackett_luce/ tests/ examples/
	mypy plackett_luce/

format:
	black plackett_luce/ tests/ examples/
	isort plackett_luce/ tests/ examples/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:
	cd docs && make html

publish: clean
	python setup.py sdist bdist_wheel
	twine upload dist/*
