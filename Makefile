.PHONY: install install-api run run-api test test-backend lint clean

install:
	pip install -r requirements.txt

install-api:
	pip install -r requirements.txt
	pip install -r backend/requirements.txt

install-dev:
	pip install -r requirements.txt -r backend/requirements.txt pytest ruff

run:
	streamlit run app.py

run-api:
	python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

test:
	python -m pytest tests/ -v

test-backend:
	python -m pytest backend/tests/ -v

lint:
	ruff check .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
