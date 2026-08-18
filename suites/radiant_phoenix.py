"""Suite: 'radiant_phoenix'.

Qwen Vision-Language Model Joint vs Sequential Bottleneck Suite (Anchor C).
Step 1: Qwen2.5-VL-8B sequential vs joint bottleneck.
Step 2: Qwen3-VL-4B sequential vs joint bottleneck.
"""
import sys
from src.utils.suite_engine import Suite, SuiteStep

SUITE = Suite(
    name="radiant_phoenix",
    title="Qwen VLM Joint Bottleneck Suite (Anchor C)",
    desc="Systematic comparison of sequential (emotion, emotion_mlp) vs end-to-end joint Stage-1 bottleneck (emotion_joint) across Qwen Vision-Language backbones (8B & 4B) under Anchor Variant C across support sizes (n=10,25,50,100, seeds 0,1,2).",
    steps=[
        SuiteStep(
            id=1,
            codename="1_joint-qwen8b",
            folder="1_joint-qwen8b",
            title="Qwen2.5-VL-8B Joint vs Sequential Bottleneck (Anchor C)",
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
        SuiteStep(
            id=2,
            codename="2_joint-qwen4b",
            folder="2_joint-qwen4b",
            title="Qwen3-VL-4B Joint vs Sequential Bottleneck (Anchor C)",
            desc="Evaluates sequential vs joint bottleneck on Qwen3-VL-4B (Layer 15) across n=10,25,50,100 with seeds 0,1,2 under Anchor C.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "qwen4b",
                "--mediators", "emotion,emotion_mlp,emotion_joint",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "C"
            ]
        ),
    ]
)
