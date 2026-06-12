#!/usr/bin/env python3
"""mlxctl — pull and serve MLX models locally for Docker Desktop containers.

Dependencies are declared in pyproject.toml and pinned in uv.lock.
Run via:  uv run python mlxctl.py <command>   (uv builds/syncs the env first).

Models run on the host (Metal GPU is not available inside containers).
Containers reach the server at http://host.docker.internal:<port>/v1.
"""

from __future__ import annotations  # keep `x | None` hints from crashing on older interpreters

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def _load_dotenv() -> None:
    """Load a local .env (HF_TOKEN and any MLXCTL_* settings) before config is read.
    Best-effort: python-dotenv may be absent under a bare interpreter, and this
    module is imported before the dependency check runs. A variable already set in
    the real shell environment always wins — load_dotenv does not override it."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


_load_dotenv()  # must run before the config block below reads os.environ


# ---- configuration --------------------------------------------------------
# Every setting is an env var with the old hard-coded value as its default, so
# nothing changes unless you set one. Precedence, lowest to highest:
#   built-in default  <  .env file  <  shell environment  <  explicit CLI flag
# (CLI flags win because argparse defaults are seeded from these values and an
#  explicit flag overrides its default.)
def _env(key: str, default: str) -> str:
    """String env var; empty/unset falls back to default."""
    return os.environ.get(key) or default


def _env_int(key: str, default: int) -> int:
    """Integer env var; unset falls back to default, a bad value warns and falls back."""
    raw = os.environ.get(key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"warning: {key}={raw!r} is not an integer; using {default}", file=sys.stderr)
        return default


STATE_DIR = Path(_env("MLXCTL_STATE_DIR", str(Path.home() / ".mlxctl"))).expanduser()
PID_FILE = STATE_DIR / "server.pid"
INFO_FILE = STATE_DIR / "server.json"
LOG_FILE = STATE_DIR / "server.log"

DEFAULT_PORT = _env_int("MLXCTL_PORT", 8080)
DEFAULT_HOST = _env("MLXCTL_HOST", "0.0.0.0")
# 32768 made a single runaway reply grow one KV cache to ~6-8 GB on a 32B model.
# 8192 is a sane *default* cap; clients can still ask for more per request.
DEFAULT_MAX_TOKENS = _env_int("MLXCTL_MAX_TOKENS", 8192)
DEFAULT_MODEL = _env("MLXCTL_MODEL", "qwen")
HEALTH_TIMEOUT = _env_int("MLXCTL_HEALTH_TIMEOUT", 300)  # secs to wait for model load

# Memory guards for a 48 GB M4 Pro. Weights are ~18 GB and macOS only wires down
# ~75% of RAM (~36 GB) for Metal, leaving ~18 GB. With no ceiling the prompt cache
# (multiple resident conversation KV caches) and request batching can exceed that
# and the server aborts with an out-of-memory error. These flags bound it.
# mlx-lm's own defaults are far higher: unbounded bytes, 10 caches, 32/8 concurrency.
DEFAULT_PROMPT_CACHE_BYTES = _env("MLXCTL_PROMPT_CACHE_BYTES", "10G")  # total KV ceiling
DEFAULT_PROMPT_CACHE_SIZE = _env_int("MLXCTL_PROMPT_CACHE_SIZE", 4)    # caches kept resident
DEFAULT_DECODE_CONCURRENCY = _env_int("MLXCTL_DECODE_CONCURRENCY", 8)  # parallel decodes
DEFAULT_PROMPT_CONCURRENCY = _env_int("MLXCTL_PROMPT_CONCURRENCY", 2)  # parallel prefills

# 4-bit mlx-community quant sized for a 48 GB M4 Pro.
# "abliterated" = refusal direction removed (won't decline tasking).
# Intended for authorized security work; you own how you use it.
_BUILTIN_NICKNAMES = {
    "qwen": "mlx-community/Qwen2.5-Coder-32B-Instruct-abliterated-4bit",  # 32B coder, abliterated
}


def _load_nicknames() -> dict[str, str]:
    """Built-in nickname table, optionally merged with a user JSON file so the list
    is editable without touching code. Path: $MLXCTL_NICKNAMES_FILE, else
    <state-dir>/models.json. File entries override built-ins.

    A flat name->repo JSON object is deliberately chosen over SQLite: it's a handful
    of static strings, stays hand-editable and diffable, needs no schema or driver,
    and survives the network hops the same way the model cache does."""
    nicks = dict(_BUILTIN_NICKNAMES)
    path = Path(_env("MLXCTL_NICKNAMES_FILE", str(STATE_DIR / "models.json"))).expanduser()
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, OSError):
        return nicks
    except json.JSONDecodeError as e:
        print(f"warning: ignoring {path}: invalid JSON ({e})", file=sys.stderr)
        return nicks
    if isinstance(data, dict):
        nicks.update({str(k): str(v) for k, v in data.items()})
    else:
        print(f"warning: ignoring {path}: expected a JSON object of nickname -> repo id",
              file=sys.stderr)
    return nicks


NICKNAMES = _load_nicknames()
REVERSE_NICKNAMES = {v: k for k, v in NICKNAMES.items()}


def resolve(name: str) -> str:
    """Nickname or full HF repo id -> repo id."""
    return NICKNAMES.get(name, name)


def display_name(repo_id: str) -> str:
    nick = REVERSE_NICKNAMES.get(repo_id)
    return f"{nick} ({repo_id})" if nick else repo_id


def hf_token() -> str | None:
    """Hugging Face token from the environment, for higher download rate limits.
    Checks HF_TOKEN first, then huggingface_hub's own HUGGING_FACE_HUB_TOKEN."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def cached_path(repo_id: str) -> str | None:
    """Return local snapshot path if fully cached, else None. No network."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(repo_id, local_files_only=True)
    except (LocalEntryNotFoundError, FileNotFoundError, OSError):
        return None


def read_pid() -> int | None:
    """Live server pid, or None. Cleans up stale files."""
    try:
        pid = int(PID_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    try:
        os.kill(pid, 0)  # liveness probe, sends no signal
        return pid
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        INFO_FILE.unlink(missing_ok=True)
        return None
    except PermissionError:
        return pid


def read_info() -> dict:
    try:
        return json.loads(INFO_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def server_supported_flags() -> set[str]:
    """Long-option flags the installed mlx_lm.server actually accepts.

    pyproject only floors the version (mlx-lm>=0.24.0) and there's no lockfile yet,
    so the memory-guard flags may or may not exist depending on what uv resolved.
    Probe `--help` once and pass only what's supported (argparse prints help and
    exits before loading any model, so this is cheap and never touches the network).
    Returns an empty set on any failure, in which case no guard flags are passed."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "mlx_lm", "server", "--help"],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        ).stdout
    except Exception:
        return set()
    return {tok.rstrip(",") for tok in out.split() if tok.startswith("--")}


def health_ok(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------- commands


def cmd_pull(args):
    repo_id = resolve(args.model)
    token = hf_token()
    auth = "authenticated" if token else "anonymous — set HF_TOKEN to avoid throttling"
    print(f"Pulling {display_name(repo_id)} ... ({auth})")
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id, token=token)
    print(f"Cached at {path}")


def cmd_serve(args):
    repo_id = resolve(args.model)
    local_path = cached_path(repo_id)
    if local_path is None:
        sys.exit(
            f"Model not cached: {display_name(repo_id)}\n"
            f"Run:  mlxctl pull {args.model}\n"
            f"(serve never touches the network, so the model must be cached first)"
        )

    # Stop-and-swap: at most one model resident.
    pid = read_pid()
    if pid:
        old = read_info().get("model", f"pid {pid}")
        print(f"Stopping running server ({old}) ...")
        _stop(pid)

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mlx_lm", "server",
        "--model", local_path,
        "--host", args.host,
        "--port", str(args.port),
        "--max-tokens", str(args.max_tokens),
    ]

    # Append memory-guard flags, but only the ones this mlx-lm version understands —
    # an unknown flag would make the server exit immediately with "unrecognized
    # arguments". A None value means "leave mlx-lm's own default in place".
    supported = server_supported_flags()
    guards = [
        ("--prompt-cache-bytes", args.prompt_cache_bytes),
        ("--prompt-cache-size", args.prompt_cache_size),
        ("--decode-concurrency", args.decode_concurrency),
        ("--prompt-concurrency", args.prompt_concurrency),
    ]
    unsupported = []
    for flag, val in guards:
        if val is None:
            continue
        if flag in supported:
            cmd += [flag, str(val)]
        elif supported:  # only warn if the probe actually returned a flag list
            unsupported.append(flag)
    if unsupported:
        print(f"note: installed mlx-lm doesn't support {', '.join(unsupported)} — "
              f"upgrade for full memory guards (uv lock --upgrade-package mlx-lm)")

    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

    if args.foreground:
        print(f"Serving {display_name(repo_id)} on {args.host}:{args.port} (foreground, Ctrl-C to stop)")
        sys.exit(subprocess.call(cmd, env=env))

    log = open(LOG_FILE, "ab")
    log.write(f"\n--- mlxctl serve {repo_id} @ {datetime.now().isoformat()} ---\n".encode())
    proc = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True, env=env,
    )
    PID_FILE.write_text(str(proc.pid))
    INFO_FILE.write_text(json.dumps({
        "model": REVERSE_NICKNAMES.get(repo_id, repo_id),
        "repo": repo_id,
        "host": args.host,
        "port": args.port,
        "pid": proc.pid,
        "started": datetime.now(timezone.utc).isoformat(),
    }, indent=2))

    print(f"Loading {display_name(repo_id)} (pid {proc.pid}) ...", end="", flush=True)
    deadline = time.monotonic() + HEALTH_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            print("\nServer exited during startup. Last log lines:")
            _tail_log(15)
            PID_FILE.unlink(missing_ok=True)
            INFO_FILE.unlink(missing_ok=True)
            sys.exit(1)
        if health_ok(args.port):
            print(" ready.")
            print(f"  local:     http://localhost:{args.port}/v1")
            print(f"  docker:    http://host.docker.internal:{args.port}/v1")
            print(f"  stop:      mlxctl stop    logs: mlxctl logs -f")
            return
        print(".", end="", flush=True)
        time.sleep(1)
    print(f"\nNot healthy after {HEALTH_TIMEOUT}s; it may still be loading. Check: mlxctl logs -f")


def _stop(pid: int, timeout: float = 10.0):
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.2)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    PID_FILE.unlink(missing_ok=True)
    INFO_FILE.unlink(missing_ok=True)


def cmd_stop(args):
    pid = read_pid()
    if not pid:
        print("No server running.")
        return
    model = read_info().get("model", f"pid {pid}")
    _stop(pid)
    print(f"Stopped {model}.")


def cmd_status(args):
    pid = read_pid()
    if not pid:
        print("Status: stopped")
        return
    info = read_info()
    port = info.get("port", DEFAULT_PORT)
    uptime = ""
    if started := info.get("started"):
        secs = int((datetime.now(timezone.utc) - datetime.fromisoformat(started)).total_seconds())
        uptime = f"{secs // 3600}h{(secs % 3600) // 60:02d}m"
    print(f"Status: running (pid {pid}, up {uptime})")
    print(f"Model:  {info.get('model', '?')}  [{info.get('repo', '?')}]")
    print(f"Health: {'ok' if health_ok(port) else 'NOT RESPONDING'}")
    print(f"  local:   http://localhost:{port}/v1")
    print(f"  docker:  http://host.docker.internal:{port}/v1")


def _tail_log(n: int):
    try:
        lines = LOG_FILE.read_text(errors="replace").splitlines()
        print("\n".join(lines[-n:]))
    except FileNotFoundError:
        print("(no log file)")


def cmd_logs(args):
    if not args.follow:
        _tail_log(args.lines)
        return
    _tail_log(args.lines)
    try:
        with open(LOG_FILE, "rb") as f:
            f.seek(0, os.SEEK_END)
            while True:
                if chunk := f.read():
                    sys.stdout.write(chunk.decode(errors="replace"))
                    sys.stdout.flush()
                else:
                    time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        print("(no log file)")


def cmd_list(args):
    from huggingface_hub import scan_cache_dir

    repos = sorted(
        (r for r in scan_cache_dir().repos if r.repo_type == "model"),
        key=lambda r: r.repo_id,
    )
    if not repos:
        print("No models cached. Try: mlxctl pull qwen")
        return
    serving = read_info().get("repo") if read_pid() else None
    for r in repos:
        nick = REVERSE_NICKNAMES.get(r.repo_id, "")
        mark = " *serving*" if r.repo_id == serving else ""
        print(f"{r.size_on_disk / 1e9:7.1f} GB  {r.repo_id}"
              + (f"  [{nick}]" if nick else "") + mark)


def cmd_rm(args):
    repo_id = resolve(args.model)
    from huggingface_hub import scan_cache_dir

    info = scan_cache_dir()
    repo = next((r for r in info.repos if r.repo_id == repo_id), None)
    if repo is None:
        sys.exit(f"Not cached: {repo_id}")
    if read_pid() and read_info().get("repo") == repo_id:
        sys.exit("That model is currently being served. Run `mlxctl stop` first.")
    revs = [rev.commit_hash for rev in repo.revisions]
    strategy = info.delete_revisions(*revs)
    strategy.execute()
    print(f"Removed {display_name(repo_id)} ({repo.size_on_disk / 1e9:.1f} GB freed)")


def cmd_nicknames(args):
    w = max(map(len, NICKNAMES))
    for nick, repo in NICKNAMES.items():
        d = " (default)" if nick == DEFAULT_MODEL else ""
        print(f"{nick:<{w}}  {repo}{d}")


def _check_environment():
    """Fail loudly and helpfully if deps aren't here — usually means the script
    was run with a bare interpreter instead of through uv's project env."""
    import importlib.util

    if importlib.util.find_spec("mlx_lm") is None:
        sys.exit(
            "mlx_lm is not available in this environment.\n\n"
            "Run mlxctl through uv so it loads the locked project dependencies:\n\n"
            "    uv run python mlxctl.py serve <model>\n\n"
            "(plain `python mlxctl.py` uses a bare interpreter with no deps;\n"
            " `uv sync` once will also populate the env.)\n"
        )


def main():
    _check_environment()  # .env was already loaded at import, before the config block
    p = argparse.ArgumentParser(
        prog="mlxctl",
        description="Pull and serve MLX models locally for Docker Desktop containers.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("pull", help="download a model (nickname or HF repo id)")
    sp.add_argument("model")
    sp.set_defaults(fn=cmd_pull)

    sp = sub.add_parser("serve", help="serve a cached model in the background (stop-and-swap)")
    sp.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    sp.add_argument("--port", type=int, default=DEFAULT_PORT)
    sp.add_argument("--host", default=DEFAULT_HOST,
                    help="bind address; 0.0.0.0 required for host.docker.internal (default)")
    sp.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help=f"default reply cap (mlx-lm's own default is 512; ours is {DEFAULT_MAX_TOKENS})")
    sp.add_argument("--prompt-cache-bytes", default=DEFAULT_PROMPT_CACHE_BYTES,
                    help=f"hard ceiling on total KV-cache memory, e.g. 10G "
                         f"(default {DEFAULT_PROMPT_CACHE_BYTES}; mlx-lm default: unbounded)")
    sp.add_argument("--prompt-cache-size", type=int, default=DEFAULT_PROMPT_CACHE_SIZE,
                    help=f"distinct conversation KV caches kept resident "
                         f"(default {DEFAULT_PROMPT_CACHE_SIZE}; mlx-lm default: 10)")
    sp.add_argument("--decode-concurrency", type=int, default=DEFAULT_DECODE_CONCURRENCY,
                    help=f"parallel decodes when batching "
                         f"(default {DEFAULT_DECODE_CONCURRENCY}; mlx-lm default: 32)")
    sp.add_argument("--prompt-concurrency", type=int, default=DEFAULT_PROMPT_CONCURRENCY,
                    help=f"parallel prefills when batching "
                         f"(default {DEFAULT_PROMPT_CONCURRENCY}; mlx-lm default: 8)")
    sp.add_argument("--foreground", action="store_true", help="run attached, for debugging")
    sp.set_defaults(fn=cmd_serve)

    sp = sub.add_parser("stop", help="stop the running server")
    sp.set_defaults(fn=cmd_stop)

    sp = sub.add_parser("status", help="show server state, model, endpoints, health")
    sp.set_defaults(fn=cmd_status)

    sp = sub.add_parser("logs", help="show server log")
    sp.add_argument("-f", "--follow", action="store_true")
    sp.add_argument("-n", "--lines", type=int, default=40)
    sp.set_defaults(fn=cmd_logs)

    sp = sub.add_parser("list", help="list cached models")
    sp.set_defaults(fn=cmd_list)

    sp = sub.add_parser("rm", help="delete a cached model")
    sp.add_argument("model")
    sp.set_defaults(fn=cmd_rm)

    sp = sub.add_parser("nicknames", help="show built-in nicknames")
    sp.set_defaults(fn=cmd_nicknames)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
