FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
RUN pip install -e .

COPY artifacts /app/artifacts
EXPOSE 8000
CMD ["uvicorn", "streaming_quality.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
