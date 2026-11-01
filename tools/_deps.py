"""Installs missing Map Lab requirements into the current interpreter (the project's .venv)."""
from __future__ import annotations

import importlib.util
import subprocess
import sys

NEEDED = {"scipy": "scipy>=1.11,<2", "pytest": "pytest>=8.3,<9", "httpx": "httpx>=0.27,<1"}
if sys.platform == "win32":
    NEEDED["pygrabber"] = "pygrabber>=0.2"


def ensure(extra: tuple[str, ...] = ()) -> None:
    wanted = {k: v for k, v in NEEDED.items() if k in extra or k in ("scipy", "pygrabber")}
    missing = [spec for mod, spec in wanted.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
