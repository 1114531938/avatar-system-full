from __future__ import annotations

import glob
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _append_log(log_file: Optional[str], text: str) -> None:
    if not log_file:
        return
    _ensure_parent(log_file)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text)


def run_bash(command: str, log_file: Optional[str] = None, check: bool = True) -> Tuple[int, str, str]:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _append_log(log_file, f"\n\n========== [{timestamp}] ==========\n$ {command}\n\n")

    proc = subprocess.Popen(
        ["bash", "-lc", command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    output_chunks = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        output_chunks.append(line)
        _append_log(log_file, line)
    proc.wait()
    output = "".join(output_chunks)
    _append_log(log_file, f"\n[exit code: {proc.returncode}]\n")

    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with code {proc.returncode}\n"
            f"Command: {command}\n\n"
            f"OUTPUT:\n{output}"
        )

    return proc.returncode, output, output


def build_apptainer_exec_command(command: str, image: str) -> str:
    q = shlex.quote
    flags = os.environ.get("APPTAINER_FLAGS", "--nv")
    thread_limited_command = (
        "export OMP_NUM_THREADS=1; "
        "export OPENBLAS_NUM_THREADS=1; "
        "export MKL_NUM_THREADS=1; "
        "export NUMEXPR_NUM_THREADS=1; "
        "export TOKENIZERS_PARALLELISM=false; "
        f"{command}"
    )
    return (
        f"apptainer exec {flags} "
        "-B /scratch:/scratch,/home/svu:/home/svu "
        f"{q(image)} bash -lc {q(thread_limited_command)}"
    )


def run_bash_in_container(
    command: str,
    image: str,
    log_file: Optional[str] = None,
    check: bool = True,
) -> Tuple[int, str, str]:
    return run_bash(build_apptainer_exec_command(command, image), log_file, check)


def run_bash_detached(command: str, log_file: Optional[str] = None) -> int:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if log_file:
        _ensure_parent(log_file)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n========== [{timestamp}] DETACHED ==========\n$ {command}\n\n")

    log_handle = open(log_file, "a", encoding="utf-8") if log_file else open(os.devnull, "a")

    proc = subprocess.Popen(
        ["bash", "-lc", command],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    return proc.pid


def latest_match(pattern: str) -> Optional[str]:
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def first_existing(*paths: str) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def require_file(path: str, desc: str = "file") -> None:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Missing {desc}: {path}")
