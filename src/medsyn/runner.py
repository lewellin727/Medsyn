"""Placeholder runner for the initial medsyn scaffold."""

from __future__ import annotations


def run_placeholder(config: dict) -> dict:
    """Return a minimal structured output until the real method is implemented."""
    return {
        "sysname": config["sysname"],
        "dataset": config["run"]["dataset"],
        "status": "placeholder",
        "records": [],
    }
