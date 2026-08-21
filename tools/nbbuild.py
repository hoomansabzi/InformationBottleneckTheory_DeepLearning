"""Tiny helper for generating the project notebooks from plain Python sources.

Writing ``.ipynb`` JSON by hand is unpleasant and hard to review.  Each notebook
in ``notebooks/`` is therefore generated from a script in ``tools/`` that
declares its cells as a list, which keeps the prose and code under normal
version control and makes regeneration reproducible.

Usage from a generator script::

    from nbbuild import md, code, build
    build("notebooks/01_minimal_model.ipynb", [md("# Title"), code("1 + 1")])
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KERNEL = "ibdl"


def md(source: str) -> dict:
    """A markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip("\n")}


def code(source: str) -> dict:
    """A code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n"),
    }


def build(path: str | Path, cells: list[dict], title: str | None = None) -> Path:
    """Write a notebook to ``path``."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (IB project)",
                "language": "python",
                "name": KERNEL,
            },
            "language_info": {"name": "python", "version": "3.12"},
            **({"title": title} if title else {}),
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {path.relative_to(PROJECT_ROOT)}  ({len(cells)} cells)")
    return path


def execute(path: str | Path, timeout: int = 3600) -> None:
    """Execute a notebook in place, populating its outputs."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    cmd = [
        str(PROJECT_ROOT / ".venv/bin/jupyter"),
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        f"--ExecutePreprocessor.kernel_name={KERNEL}",
        str(path),
    ]
    print(f"executing {path.name} ...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-6000:])
        sys.stderr.write(result.stderr[-6000:])
        raise SystemExit(f"execution of {path.name} failed")
    print(f"executed {path.name}")


#: Preamble shared by every notebook: puts the project root on ``sys.path`` so
#: ``import ibdl`` works regardless of where Jupyter was launched from.
PREAMBLE = """
import sys, pathlib
ROOT = pathlib.Path.cwd()
if not (ROOT / "ibdl").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from ibdl import plotting
plotting.use_paper_style()
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True)

print("project root:", ROOT)
"""
