"""Experiment Suite: 'crimson_falcon' (Generated via coolname).

Evaluates sequential vs joint Stage-1 bottleneck on Qwen2.5-VL-8B with residual Anchor Variant C.
"""
import sys
from src.utils.suite_engine import Suite, SuiteStep

SUITE = Suite(
    name="crimson_falcon",
    title="Qwen2.5-VL-8B Joint Bottleneck Sweep (Anchor C)",
    desc="Sample-efficiency sweep (n=10,25,50,100) comparing sequential (emotion, emotion_mlp) vs joint Stage-1 bottleneck (emotion_joint) on Qwen2.5-VL-8B under Anchor Variant C.",
    steps=[
        SuiteStep(
            id=1,
            codename="joint-c",
            folder="1_joint-c",
            title="Qwen8B Joint vs Sequential (Anchor C)",
            desc="Evaluates sequential vs joint bottleneck on Qwen2.5-VL-8B across n=10,25,50,100 with seeds 0,1,2 under Anchor C.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "qwen8b",
                "--mediators", "emotion,emotion_mlp,emotion_joint",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "C"
            ]
        ),
    ]
)
