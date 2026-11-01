"""Runs the test suite with the project's interpreter (VS Code: open this file and press Run)."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from tools._deps import ensure

    ensure(("pytest", "httpx"))
    import pytest

    os.chdir(ROOT)
    sys.exit(pytest.main(["-q", "tests"]))
