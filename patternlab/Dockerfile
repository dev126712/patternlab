FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml ./
RUN uv sync --no-install-project --extra agent
COPY . .
RUN uv sync --extra agent
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "patternlab.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
