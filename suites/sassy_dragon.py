"""Suite: 'sassy_dragon'.

Multimodal Backbone Distributional & Joint Bottleneck Suite (Anchor C).
Round 1: Distributional Stage-1 (mean, sd, histogram) across backbones (clip_ft, clip_ft_emo, qwen4b).
Round 2: End-to-end Joint vs Sequential bottleneck across backbones (clip_ft, clip_ft_emo).
"""
import sys
from src.utils.suite_engine import Suite, SuiteStep

SUITE = Suite(
    name="sassy_dragon",
    title="Multimodal Distributional & Joint Bottleneck Suite (Anchor C)",
    desc="Two-round systematic sweep under Anchor C across sample sizes (n=10,25,50,100, seeds 0,1,2): Round 1 explores distributional Stage-1 emotion representations; Round 2 evaluates joint vs sequential bottleneck.",
    steps=[
        SuiteStep(
            id=1,
            codename="1A_dist-clip-ft",
            folder="1A_dist-clip-ft",
            title="[1A] CLIP-ft (score) Distributional Bottleneck (Anchor C)",
            desc="Evaluates emotion distributional representations (mean, sd, histogram) on CLIP-ft (score) under Anchor C across n=10,25,50,100 with seeds 0,1,2.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "clip_ft",
                "--mediators", "emotion,emotion_sd,emotion_hist",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "C"
            ]
        ),
        SuiteStep(
            id=2,
            codename="1B_dist-clip-ft-emo",
            folder="1B_dist-clip-ft-emo",
            title="[1B] CLIP-ft (emotion) Distributional Bottleneck (Anchor C)",
            desc="Evaluates emotion distributional representations on CLIP-ft (emotion) under Anchor C across n=10,25,50,100 with seeds 0,1,2.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "clip_ft_emo",
                "--mediators", "emotion,emotion_sd,emotion_hist",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "C"
            ]
        ),
        SuiteStep(
            id=3,
            codename="1C_dist-qwen4b",
            folder="1C_dist-qwen4b",
            title="[1C] Qwen3-4B Distributional Bottleneck (Anchor C)",
            desc="Evaluates emotion distributional representations on Qwen3-VL-4B (Layer 15) under Anchor C across n=10,25,50,100 with seeds 0,1,2.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "qwen4b",
                "--mediators", "emotion,emotion_sd,emotion_hist",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "C"
            ]
        ),
        SuiteStep(
            id=4,
            codename="2A_joint-clip-ft",
            folder="2A_joint-clip-ft",
            title="[2A] CLIP-ft (score) Sequential vs Joint Bottleneck (Anchor C)",
            desc="Compares sequential Stage-1 (ridge, MLP) vs joint bottleneck on CLIP-ft (score) under Anchor C across n=10,25,50,100 with seeds 0,1,2.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "clip_ft",
                "--mediators", "emotion,emotion_mlp,emotion_joint",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "C"
            ]
        ),
        SuiteStep(
            id=5,
            codename="2B_joint-clip-ft-emo",
            folder="2B_joint-clip-ft-emo",
            title="[2B] CLIP-ft (emotion) Sequential vs Joint Bottleneck (Anchor C)",
            desc="Compares sequential Stage-1 vs joint bottleneck on CLIP-ft (emotion) under Anchor C across n=10,25,50,100 with seeds 0,1,2.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "clip_ft_emo",
                "--mediators", "emotion,emotion_mlp,emotion_joint",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "C"
            ]
        ),
    ]
)
