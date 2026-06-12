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

Serve flags: `--port 8080`, `--host 0.0.0.0`, `--max-tokens 8192`.

Memory guards (defaults sized for a 48 GB M4 Pro; lower if you still hit OOM, raise
if you have headroom): `--prompt-cache-bytes 10G` (hard ceiling on total KV-cache
memory), `--prompt-cache-size 4` (distinct conversation caches kept resident),
`--decode-concurrency 8` and `--prompt-concurrency 2` (bound batch memory spikes).
These are only passed if the installed `mlx-lm` supports them — older versions fall
back to their own (much higher) defaults with a printed note.

## Configuration

Every default is an environment variable; the built-in value is used unless you set
one. Resolution order, lowest to highest priority:

```
built-in default  <  .env file  <  shell environment  <  explicit CLI flag
```

So `MLXCTL_PORT=9000 mlxctl serve` serves on 9000, but `mlxctl serve --port 8080`
still wins over the env var. Settings live in `.env` (copy from `.env.example`):

| Variable | Default | What it sets |
|---|---|---|
| `MLXCTL_MODEL` | `qwen` | Default model for `serve` with no argument |
| `MLXCTL_HOST` | `0.0.0.0` | Bind address (`0.0.0.0` needed for `host.docker.internal`) |
| `MLXCTL_PORT` | `8080` | Server port |
| `MLXCTL_MAX_TOKENS` | `8192` | Default reply cap when a request omits `max_tokens` |
| `MLXCTL_PROMPT_CACHE_BYTES` | `10G` | Hard ceiling on total KV-cache memory |
| `MLXCTL_PROMPT_CACHE_SIZE` | `4` | Distinct conversation caches kept resident |
| `MLXCTL_DECODE_CONCURRENCY` | `8` | Parallel decodes when batching |
| `MLXCTL_PROMPT_CONCURRENCY` | `2` | Parallel prefills when batching |
| `MLXCTL_HEALTH_TIMEOUT` | `300` | Seconds `serve` waits for the model to load |
| `MLXCTL_STATE_DIR` | `~/.mlxctl` | Where pid/info/log/`models.json` live |
| `MLXCTL_NICKNAMES_FILE` | `<state-dir>/models.json` | Extra nicknames, merged over built-ins |

### Custom nicknames

The built-in nickname table is just a starting point. Drop a JSON object at
`~/.mlxctl/models.json` (or point `MLXCTL_NICKNAMES_FILE` elsewhere) to add your own;
entries override built-ins by name:

```json
{
  "qwen-small": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
  "llama": "mlx-community/Llama-3.3-70B-Instruct-4bit"
}
```

(JSON rather than a database on purpose — it's a handful of static `name -> repo`
strings, so a hand-editable, diffable file beats a schema and a driver.)

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
- **`--max-tokens`** defaults to 8192. mlx-lm's server otherwise caps replies at ~500
  tokens, which is too low — but the old 32768 default let a single runaway reply grow
  one KV cache to ~6-8 GB on a 32B model and tip the machine into OOM. 8192 is a safe
  default; raise it per request when you actually need a long completion.
- **Memory guards.** On a 48 GB machine the ~18 GB of weights leave only ~18 GB after
  macOS's Metal wired-memory limit, and mlx-lm's defaults (unbounded prompt-cache
  bytes, 10 resident caches, 32-way decode batching) can blow past that and abort the
  server with an out-of-memory error — especially with several containers sending
  different prompts. `serve` now caps these (`--prompt-cache-bytes`,
  `--prompt-cache-size`, `--decode-concurrency`, `--prompt-concurrency`). If you raise
  the macOS limit with `sudo sysctl iogpu.wired_limit_mb=<mb>` you can relax them.
- **Binding `0.0.0.0`** is required for `host.docker.internal` to work — but the
  mlx-lm server has only basic security checks, so don't expose the port beyond your
  machine/tailnet. No reverse proxy, no public interfaces.
- State lives in `~/.mlxctl/` (`server.pid`, `server.json`, `server.log`); model
  weights live in the standard Hugging Face cache (`~/.cache/huggingface/hub`).
