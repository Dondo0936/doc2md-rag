.PHONY: install run test lint clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt pytest ruff

run:
	streamlit run app.py

test:
	cd .. && python -m pytest doc2md-rag/tests/ -v

lint:
	ruff check .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
