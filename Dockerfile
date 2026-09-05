FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

# Default: MCP stdio for agents. Override for UI:
#   docker run ... streamlit run app.py --server.headless true --server.port 8501
EXPOSE 8501
CMD ["doc2md-rag", "mcp"]