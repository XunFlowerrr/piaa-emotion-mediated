"""Dedicated Runner for Suite 'crimson_falcon'."""
from suites.crimson_falcon import SUITE
from src.utils.suite_engine import run_suite_cli

if __name__ == "__main__":
    run_suite_cli(SUITE)
