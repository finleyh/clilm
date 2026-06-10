#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mlx-lm>=0.24.0",
#   "huggingface_hub>=0.30.0",
# ]
# ///
"""mlxctl — pull and serve MLX models locally for Docker Desktop containers.

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

STATE_DIR = Path.home() / ".mlxctl"
PID_FILE = STATE_DIR / "server.pid"
INFO_FILE = STATE_DIR / "server.json"
LOG_FILE = STATE_DIR / "server.log"

DEFAULT_PORT = 8080
DEFAULT_HOST = "0.0.0.0"
DEFAULT_MAX_TOKENS = 32768
DEFAULT_MODEL = "qwen3-30b"
HEALTH_TIMEOUT = 300  # seconds to wait for model load before giving up

# 4-bit mlx-community quants sized for a 48 GB M4 Pro.
NICKNAMES = {
    "qwen3-30b": "mlx-community/Qwen3-30B-A3B-4bit",
    "qwen3-14b": "mlx-community/Qwen3-14B-4bit",
    "gemma3-12b": "mlx-community/gemma-3-12b-it-4bit",
    "llama3.1-8b": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "llama3.2-3b": "mlx-community/Llama-3.2-3B-Instruct-4bit",
}
REVERSE_NICKNAMES = {v: k for k, v in NICKNAMES.items()}


def resolve(name: str) -> str:
    """Nickname or full HF repo id -> repo id."""
    return NICKNAMES.get(name, name)


def display_name(repo_id: str) -> str:
    nick = REVERSE_NICKNAMES.get(repo_id)
    return f"{nick} ({repo_id})" if nick else repo_id


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


def health_ok(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------- commands


def cmd_pull(args):
    repo_id = resolve(args.model)
    print(f"Pulling {display_name(repo_id)} ...")
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id)
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
        print("No models cached. Try: mlxctl pull qwen3-30b")
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
    """Fail loudly and helpfully if the deps aren't here — usually means
    'python' was wrongly put in the command, so uv skipped the # /// script block."""
    import importlib.util

    if importlib.util.find_spec("mlx_lm") is None:
        sys.exit(
            "mlx_lm is not available in this environment.\n\n"
            "This almost always means you ran:   uv run python mlxctl.py ...\n"
            "Putting 'python' in the middle makes uv ignore the inline dependency\n"
            "block. Run it WITHOUT 'python':\n\n"
            "    uv run mlxctl.py serve <model>      (or  ./mlxctl.py serve <model>)\n"
        )


def main():
    _check_environment()
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
