FROM python:3.13-slim

# Minimal inference-only image: no training dependencies baked in.
RUN useradd --create-home --uid 1000 localai
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

RUN mkdir -p /app/data /app/artifacts && chown -R localai:localai /app
USER localai

ENTRYPOINT ["local-ai"]
CMD ["--help"]
