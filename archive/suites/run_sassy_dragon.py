"""Dedicated Runner for Suite 'sassy_dragon'."""
from suites.sassy_dragon import SUITE
from src.utils.suite_engine import run_suite_cli

if __name__ == "__main__":
    run_suite_cli(SUITE)
