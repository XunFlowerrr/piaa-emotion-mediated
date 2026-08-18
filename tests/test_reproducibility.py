"""Test bit-for-bit reproducibility between single-core serial and multi-core parallel runs.

Usage:
    uv run tests/test_reproducibility.py
    # or
    uv run main.py verify --parallel
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.utils.verify import check_parallel_repro


def test_parallel_vs_serial_reproducibility():
    cfg = Config()
    # Test on default backbone with seeds 0 and 1, n_train=10 across all 5 folds
    success = check_parallel_repro(cfg, backbone="clip", seeds=(0, 1), n_train=10)
    assert success, "Multi-core parallel output does not match single-core serial output!"
    print("\n✓ ALL REPRODUCIBILITY TESTS PASSED!")


if __name__ == "__main__":
    test_parallel_vs_serial_reproducibility()
