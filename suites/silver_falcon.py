"""Suite: 'silver_falcon'.

Standard Stage-2 Multimodal Backbone Baseline Sweep Suite (Anchor C).
Evaluates full standard Stage-2 controls (identity, pca, emotion, random, shuffled)
across all 5 vision backbones under Anchor Variant C (n=10,25,50,100, seeds 0,1,2).
"""
import sys
from src.utils.suite_engine import Suite, SuiteStep

SUITE = Suite(
    name="silver_falcon",
    title="Standard Stage-2 Multimodal Baseline Sweep Suite (Anchor C)",
    desc="Comprehensive Stage-2 baseline and control sweep (identity, pca, emotion, random, shuffled) across all 5 vision backbones under Anchor Variant C across support sizes (n=10,25,50,100, seeds 0,1,2).",
    steps=[
        SuiteStep(
            id=1,
            codename="1_base-clip",
            folder="1_base-clip",
            title="Standard Controls Sweep on Frozen CLIP ViT-B/32 (Anchor C)",
            desc="Full Stage-2 controls (identity, pca, emotion, random, shuffled) on Frozen CLIP under Anchor C.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "clip",
                "--mediators", "identity,pca,emotion,random,shuffled",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "C"
            ]
        ),
        SuiteStep(
            id=2,
            codename="2_base-qwen8b",
            folder="2_base-qwen8b",
            title="Standard Controls Sweep on Qwen3-VL 8B (Anchor C)",
            desc="Full Stage-2 controls (identity, pca, emotion, random, shuffled) on Qwen3-VL 8B under Anchor C.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "qwen8b",
                "--mediators", "identity,pca,emotion,random,shuffled",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "C"
            ]
        ),
        SuiteStep(
            id=3,
            codename="3_base-clip-ft",
            folder="3_base-clip-ft",
            title="Standard Controls Sweep on CLIP-ft Score (Anchor C)",
            desc="Full Stage-2 controls (identity, pca, emotion, random, shuffled) on fine-tuned CLIP (score) under Anchor C.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "clip_ft",
                "--mediators", "identity,pca,emotion,random,shuffled",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "C"
            ]
        ),
        SuiteStep(
            id=4,
            codename="4_base-clip-ft-emo",
            folder="4_base-clip-ft-emo",
            title="Standard Controls Sweep on CLIP-ft Emotion (Anchor C)",
            desc="Full Stage-2 controls (identity, pca, emotion, random, shuffled) on fine-tuned CLIP (emotion) under Anchor C.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "clip_ft_emo",
                "--mediators", "identity,pca,emotion,random,shuffled",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "C"
            ]
        ),
        SuiteStep(
            id=5,
            codename="5_base-qwen4b",
            folder="5_base-qwen4b",
            title="Standard Controls Sweep on Qwen3-VL 4B (Anchor C)",
            desc="Full Stage-2 controls (identity, pca, emotion, random, shuffled) on Qwen3-VL 4B (Layer 15) under Anchor C.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "qwen4b",
                "--mediators", "identity,pca,emotion,random,shuffled",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "C"
            ]
        ),
    ]
)
