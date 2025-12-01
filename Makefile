# Makefile for running tests in the Python package

# Run tests
test:
	pytest

# Run testes with coverage
test-coverage:
	pytest --cov --cov-report=html:htmlcov