# mlxctl

Pull and serve LLMs locally on Apple Silicon (MLX), exposed on an OpenAI-compatible
`/v1` endpoint that Docker Desktop containers can reach. Think "LM Studio's server,
but a CLI" — built for an M4 Pro side laptop that hops networks, so once a model is
cached, serving never touches the network.

## Requirements

- Apple Silicon Mac (built for an M4 Pro, 48 GB)
- [uv](https://docs.astral.sh/uv/) — `brew install uv`
- Docker Desktop (only if containers are the consumer)

This is a uv project. Dependencies are declared openly in `pyproject.toml` and
pinned in `uv.lock` — nothing hidden. No manual venv or `pip install`: `uv run`
creates the environment, installs the locked deps, and runs the script.

## Install

```sh
uv sync                      # build the env from uv.lock (first run; optional)
```

`uv.lock` is generated automatically the first time you run anything (`uv run` /
`uv sync`). Commit it for reproducible installs; regenerate after editing
dependencies in `pyproject.toml` with `uv lock`.

### Hugging Face token (avoid download throttling)

```sh
cp .env.example .env     # then paste your token into .env
```

mlxctl loads `.env` automatically (via python-dotenv), so `mlxctl pull` runs
authenticated and won't get rate-limited. `.env` is gitignored. A read-scope token
from <https://huggingface.co/settings/tokens> is enough; it also unlocks gated
repos. Any `HF_TOKEN` already exported in your shell takes precedence over `.env`.

## Quick start

Run every command as `uv run python mlxctl.py <command>` — uv syncs the locked
environment first, so `mlx_lm` is always present:

```sh
uv run python mlxctl.py pull qwen          # download (one-time, needs network)
uv run python mlxctl.py serve qwen         # serve in the background on port 8080
uv run python mlxctl.py status             # pid, model, uptime, health, endpoints
uv run python mlxctl.py stop               # unload the model, free the RAM
```

Tired of typing the prefix? Drop a one-line wrapper on your PATH:

```sh
printf '#!/bin/sh\nexec uv run --project "%s" python "%s/mlxctl.py" "$@"\n' \
  "$(pwd)" "$(pwd)" > /usr/local/bin/mlxctl && chmod +x /usr/local/bin/mlxctl
# then:  mlxctl serve qwen
```

`serve` detaches and returns once the model answers health checks. Endpoints:

| From | URL |
|---|---|
| the Mac itself | `http://localhost:8080/v1` |
| a Docker container | `http://host.docker.internal:8080/v1` |

## Commands

| Command | What it does |
|---|---|
| `pull <model>` | Download by nickname or full HF repo id |
| `serve [model]` | Serve in background (default model: `qwen`). Serving while another model runs stops it first — at most one model resident |
| `serve --foreground` | Run attached, for debugging |
| `stop` | SIGTERM the server, clean up pidfile |
| `status` | Running state, model, uptime, health ping, endpoints |
| `logs [-f] [-n N]` | Show / follow the server log (`~/.mlxctl/server.log`) |
| `list` | Cached models with sizes; marks the one being served |
| `rm <model>` | Delete a cached model (refuses if currently served) |
| `nicknames` | Show the built-in nickname table |

Serve flags: `--port 8080`, `--host 0.0.0.0`, `--max-tokens 32768`.

## Nicknames

4-bit mlx-community quant sized for 48 GB unified memory. This is an
**abliterated** build — the refusal direction has been removed, so it won't decline
tasking the way stock instruct models do. Intended for authorized security work;
you own how you use it.

| Nickname | Repo | Notes |
|---|---|---|
| `qwen` | mlx-community/Qwen2.5-Coder-32B-Instruct-abliterated-4bit | **default** — 32B coder, ~18 GB |

Anything else works by full repo id: `mlxctl pull mlx-community/SomeModel-4bit`.

> Earlier 8B picks (Daredevil, Llama-3.1) were dropped: they're Llama-3-derived and
> ship a broken stop-token config (`eos_token_id` omits `<|eot_id|>`), so they run
> past their turn and start answering themselves. Qwen uses `<|im_end|>` and stops
> correctly.

## Using from containers

The Metal GPU is **not** available inside Docker containers, so the model runs on the
host and containers call out to it.

Quick test from inside any container:

```sh
curl http://host.docker.internal:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "hello"}]}'
```

### OpenWebUI on Docker Desktop

```yaml
services:
  openwebui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    environment:
      - OPENAI_API_BASE_URL=http://host.docker.internal:8080/v1
      - OPENAI_API_KEY=local            # any non-empty string
    extra_hosts:
      - "host.docker.internal:host-gateway"   # needed on older Compose
    volumes:
      - openwebui-data:/app/backend/data
volumes:
  openwebui-data:
```

Then `docker compose up -d` and open http://localhost:3000. Switching models is
`mlxctl serve <other-nickname>` on the host — OpenWebUI keeps pointing at the same
endpoint.

## Design notes & gotchas

- **Offline by design.** `serve` only loads from the local cache (it errors and tells
  you to `pull` if the model isn't cached) and launches the server with
  `HF_HUB_OFFLINE=1`, so wifi/cellular/Tailscale hops can't break inference.
- **Daemon, not launchd.** The server detaches with a pidfile under `~/.mlxctl/`.
  There is deliberately no auto-restart: a supervisor that silently reloads a ~17 GB
  model on crash or login would pin half the RAM unasked. Models occupy memory only
  between an explicit `serve` and `stop`. (A `launchd` opt-in may come later.)
- **One model at a time.** `serve` swaps, never stacks.
- **`--max-tokens`** defaults to 32768 because mlx-lm's server otherwise caps replies
  at ~500 tokens.
- **Binding `0.0.0.0`** is required for `host.docker.internal` to work — but the
  mlx-lm server has only basic security checks, so don't expose the port beyond your
  machine/tailnet. No reverse proxy, no public interfaces.
- State lives in `~/.mlxctl/` (`server.pid`, `server.json`, `server.log`); model
  weights live in the standard Hugging Face cache (`~/.cache/huggingface/hub`).
