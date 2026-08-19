"""Dedicated Runner for Suite 'scarlet_swallow'."""
from suites.scarlet_swallow import SUITE
from src.utils.suite_engine import run_suite_cli

if __name__ == "__main__":
    run_suite_cli(SUITE)
