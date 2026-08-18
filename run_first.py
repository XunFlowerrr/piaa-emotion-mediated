"""Dedicated Runner for Suite 'first'."""
from suites.first import SUITE
from src.utils.suite_engine import run_suite_cli

if __name__ == "__main__":
    run_suite_cli(SUITE)
