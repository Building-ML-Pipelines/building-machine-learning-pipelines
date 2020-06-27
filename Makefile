.PHONY: help clean clean-pyc clean-build list test test-all coverage docs release sdist

help:
	@echo "clean - remove build artifacts"
	@echo "develop - set up dev environment"
	@echo "install-deps"
	@echo "install-pre-commit"
	@echo "setup-git"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"


clean:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

develop: setup-git install-deps install-pre-commit

install-deps:
	pip install -r requirements/requirements.txt
	pip install -r requirements/test-requirements.txt

install-pre-commit:
	pip install "pre-commit>=1.10.1,<1.11.0"

setup-git: install-pre-commit
	pre-commit install
	git config branch.autosetuprebase always

lint: install-pre-commit
	@echo "Linting Python files"
	pre-commit run -a
	@echo ""

test: develop lint
	@echo "Running Python tests"
	py.test .
	@echo ""
