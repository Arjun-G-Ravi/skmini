# Define the Python interpreter to use
PYTHON = python3

# Define the pip command
PIP = $(PYTHON) -m pip

# Default target
.PHONY: all
all: install test

# Install requirements
.PHONY: install
install:
	$(PIP) install -r requirements.txt

# Run tests
.PHONY: test
test:
	$(PYTHON) tests/unit_tests/test_logistic_regression.py
	$(PYTHON) tests/unit_tests/test_linear_regression.py

# Clean compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Help target
.PHONY: help
help:
	@echo "Usage:"
	@echo "  make          - Install requirements and run all tests"
	@echo "  make install  - Install all dependencies"
	@echo "  make test     - Run the unit tests"
	@echo "  make clean    - Clean up compiled Python files"
	@echo "  make help     - Show this help message"
