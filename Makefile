.PHONY: install run test lint clean cli mcp

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt pytest ruff
	pip install -e .

run:
	streamlit run app.py

cli:
	doc2md-rag --help

mcp:
	doc2md-rag mcp

test:
	python -m pytest tests/ -v

lint:
	ruff check .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .doc2md_kb *.egg-info dist build
