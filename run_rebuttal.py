"""Dedicated Runner for Suite 'rebuttal'."""
from suites.rebuttal import SUITE
from src.utils.suite_engine import run_suite_cli

if __name__ == "__main__":
    run_suite_cli(SUITE)
