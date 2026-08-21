"""Suite: 'demonic_bobcat'.

Qwen3-VL 8B MLP Series Rebuild Suite (Anchor C, n=100, MLP Head).
Evaluates sequential (emotion_mlp + MLP head) vs joint (emotion_joint + MLP head)
bottleneck models on Qwen3-VL 8B under Anchor Variant C across seeds 0, 1, 2.
"""
import sys
from src.utils.suite_engine import Suite, SuiteStep

SUITE = Suite(
    name="demonic_bobcat",
    title="Qwen3-VL 8B MLP Series Rebuild Suite (Anchor C, n=100, MLP Head)",
    desc="Evaluates sequential (emotion_mlp) vs joint (emotion_joint) bottleneck models with MLP head on Qwen3-VL 8B under Anchor C at n=100 across seeds 0, 1, 2.",
    steps=[
        SuiteStep(
            id=1,
            codename="1_mlp-s0",
            folder="1_mlp-s0",
            title="MLP Series on Qwen3-VL 8B (Seed 0, n=100, Anchor C)",
            desc="Sequential (emotion_mlp) and Joint (emotion_joint) with MLP head on Qwen3-VL 8B at n=100, seed 0, Anchor C.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "qwen8b",
                "--mediators", "emotion_mlp,emotion_joint",
                "--heads", "mlp",
                "--n-train", "100",
                "--seed", "0",
                "--stage2", "C"
            ]
        ),
        SuiteStep(
            id=2,
            codename="2_mlp-s1",
            folder="2_mlp-s1",
            title="MLP Series on Qwen3-VL 8B (Seed 1, n=100, Anchor C)",
            desc="Sequential (emotion_mlp) and Joint (emotion_joint) with MLP head on Qwen3-VL 8B at n=100, seed 1, Anchor C.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "qwen8b",
                "--mediators", "emotion_mlp,emotion_joint",
                "--heads", "mlp",
                "--n-train", "100",
                "--seed", "1",
                "--stage2", "C"
            ]
        ),
        SuiteStep(
            id=3,
            codename="3_mlp-s2",
            folder="3_mlp-s2",
            title="MLP Series on Qwen3-VL 8B (Seed 2, n=100, Anchor C)",
            desc="Sequential (emotion_mlp) and Joint (emotion_joint) with MLP head on Qwen3-VL 8B at n=100, seed 2, Anchor C.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "qwen8b",
                "--mediators", "emotion_mlp,emotion_joint",
                "--heads", "mlp",
                "--n-train", "100",
                "--seed", "2",
                "--stage2", "C"
            ]
        ),
    ]
)
