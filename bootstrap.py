from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
GUI_REQUIREMENTS_FILE = ROOT / "requirements.txt"
SIM_REQUIREMENTS_FILE = ROOT / "requirements-sim.txt"
GUI_MARKER_FILE = VENV_DIR / ".gui-requirements.sha256"
SIM_MARKER_FILE = VENV_DIR / ".sim-requirements.sha256"
LOG_FILE = ROOT / "bootstrap.log"


def apple_quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def show_dialog(message: str, title: str = "NetPyNe Modeler") -> None:
    if sys.platform != "darwin":
        return
    script = (
        f"display dialog {apple_quote(message)} with title {apple_quote(title)} "
        'buttons {"OK"} default button "OK"'
    )
    subprocess.run(["/usr/bin/osascript", "-e", script], check=False)


def log(message: str) -> None:
    print(message)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def run_logged(command: list[str], cwd: Path | None = None) -> None:
    log("$ " + " ".join(shlex.quote(part) for part in command))
    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=cwd or ROOT,
            env=env,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def active_requirements(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def requirements_hash(path: Path) -> str:
    requirements = "\n".join(active_requirements(path))
    return hashlib.sha256(requirements.encode("utf-8")).hexdigest()


def install_requirements(path: Path, marker_path: Path, label: str) -> None:
    requirements = active_requirements(path)
    if not requirements:
        return

    current_hash = requirements_hash(path)
    installed_hash = marker_path.read_text(encoding="utf-8").strip() if marker_path.exists() else ""
    if current_hash == installed_hash:
        return

    show_dialog(f"Installing the optional {label}. This can take several minutes.")
    pip_path = VENV_DIR / "bin" / "pip"
    if not pip_path.exists():
        raise RuntimeError("Virtual environment exists, but pip is missing.")
    run_logged([str(pip_path), "install", "-r", str(path)])
    marker_path.write_text(current_hash, encoding="utf-8")


def ensure_venv(install_sim_stack: bool = True) -> None:
    python_path = VENV_DIR / "bin" / "python"

    if not python_path.exists():
        show_dialog("Creating .venv for NetPyNe Modeler.")
        run_logged([sys.executable, "-m", "venv", str(VENV_DIR)])

    install_requirements(GUI_REQUIREMENTS_FILE, GUI_MARKER_FILE, "GUI dependencies")
    if install_sim_stack:
        install_requirements(SIM_REQUIREMENTS_FILE, SIM_MARKER_FILE, "simulation stack")


def launch_app(argv: list[str]) -> int:
    python_path = VENV_DIR / "bin" / "python"
    command = [str(python_path), "-m", "netpyne_modeler", *argv]
    os.execv(command[0], command)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    install_sim_stack = True
    no_launch = False
    passthrough_args: list[str] = []

    for arg in args:
        if arg == "--install-sim-stack":
            install_sim_stack = True
        elif arg == "--skip-sim-stack":
            install_sim_stack = False
        elif arg == "--no-launch":
            no_launch = True
        else:
            passthrough_args.append(arg)

    LOG_FILE.write_text("", encoding="utf-8")
    try:
        ensure_venv(install_sim_stack=install_sim_stack)
    except Exception as exc:
        log(f"Bootstrap failed: {exc}")
        show_dialog(
            "Bootstrap failed. Inspect bootstrap.log in the project directory for details."
        )
        return 1
    if no_launch:
        return 0
    return launch_app(passthrough_args)


if __name__ == "__main__":
    raise SystemExit(main())
