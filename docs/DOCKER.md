# Docker

Status: configuration written in Milestone 1, **not yet build-tested** —
Docker was not installed on the development machine when this was
written. Test with the commands below once Docker Desktop is installed,
then update this note with the actual result.

## Design

- `cli` service: the Python CLI, built from the project `Dockerfile`.
- Ollama runs on the Mac **host**, not as a container, for Milestone 1-2.
  A container-Ollama option may be added later if it's actually tested
  and found reliable on Apple Silicon.
- `./data` and `./artifacts` are bind-mounted so files survive container
  removal and stay inspectable from the host.
- No ports are published. This is a CLI tool, not a server.
- The image installs only the base (inference-only) dependencies —
  training extras (`pip install ".[train]"`) are not baked in.

## Commands

```bash
# Build the image
docker compose build

# See the CLI help
docker compose run --rm cli --help

# Run a command (requires .env — copy .env.example first)
cp .env.example .env
docker compose run --rm cli models

# Stop and remove containers (does not touch ./data or ./artifacts,
# since those are host directories, not named volumes)
docker compose down
```

## macOS networking note

Inside the `cli` container, `localhost` refers to the container itself,
not the Mac host running Ollama. On Docker Desktop for Mac,
`host.docker.internal` resolves to the host — set in `.env`:

```
LOCAL_AI_OLLAMA_HOST=http://host.docker.internal:11434
```

This has not yet been verified end-to-end (Milestone 2, once Ollama is
installed and wired up). On Linux, `host.docker.internal` is not
guaranteed to work the same way; that will be checked in Milestone 6.

## Destructive operations

`docker compose down -v` would remove named volumes. This project
currently defines none (model storage lives on the host via Ollama, and
`./data`/`./artifacts` are bind mounts, not volumes), but if a future
milestone adds one, `-v` will delete it — only run it when you explicitly
intend to discard that volume's contents.
