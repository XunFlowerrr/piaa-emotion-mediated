"""Suite: 'emerald_tiger'.

Comprehensive Multimodal Backbone & Mediator Sweep Suite (Anchor C).
13 Flattened Steps:
- Steps 1-4: Standard Stage-2 Baseline Sweeps (qwen8b, clip_ft, clip_ft_emo, qwen4b)
- Steps 5-8: Stage-1 MLP & Joint Bottleneck Sweeps (qwen8b, clip_ft, clip_ft_emo, qwen4b)
- Steps 9-13: Stage-1 Distributional Mediators Sweeps (clip, qwen8b, clip_ft, clip_ft_emo, qwen4b)
"""
import sys
from src.utils.suite_engine import Suite, SuiteStep

SUITE = Suite(
    name="emerald_tiger",
    title="Comprehensive Multimodal Backbone & Mediator Sweep Suite (Anchor C)",
    desc="Comprehensive 13-step flattened evaluation sweep under Anchor Variant C across all 5 backbones (clip, clip_ft, clip_ft_emo, qwen4b, qwen8b) covering standard baselines (1-4), nonlinear & joint bottlenecks (5-8), and distributional mediators (9-13) across support sizes n in {10,25,50,100} with seeds 0,1,2.",
    steps=[
        # --- Group 1: Standard Stage-2 Baseline Sweeps (Anchor C) ---
        SuiteStep(
            id=1,
            codename="1_base-qwen8b",
            folder="1_base-qwen8b",
            title="Standard Stage-2 Sweep on Qwen3-VL 8B (Anchor C)",
            desc="Standard Stage-2 sweep with default mediators (identity, emotion, pca, random, shuffled) on Qwen3-VL 8B under Anchor C.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "qwen8b",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "C"
            ]
        ),
        SuiteStep(
            id=2,
            codename="2_base-clip-ft",
            folder="2_base-clip-ft",
            title="Standard Stage-2 Sweep on CLIP-ft Score (Anchor C)",
            desc="Standard Stage-2 sweep with default mediators on fine-tuned CLIP (overall aesthetic score) under Anchor C.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "clip_ft",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "C"
            ]
        ),
        SuiteStep(
            id=3,
            codename="3_base-clip-ft-emo",
            folder="3_base-clip-ft-emo",
            title="Standard Stage-2 Sweep on CLIP-ft Emotion (Anchor C)",
            desc="Standard Stage-2 sweep with default mediators on fine-tuned CLIP (emotion supervision) under Anchor C.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "clip_ft_emo",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "C"
            ]
        ),
        SuiteStep(
            id=4,
            codename="4_base-qwen4b",
            folder="4_base-qwen4b",
            title="Standard Stage-2 Sweep on Qwen3-VL 4B (Anchor C)",
            desc="Standard Stage-2 sweep with default mediators on Qwen3-VL 4B (Layer 15) under Anchor C.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "qwen4b",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "C"
            ]
        ),

        # --- Group 2: Stage-1 MLP & Joint Bottleneck Sweeps (Anchor C) ---
        SuiteStep(
            id=5,
            codename="5_joint-qwen8b",
            folder="5_joint-qwen8b",
            title="Sequential vs Joint Bottleneck on Qwen3-VL 8B (Anchor C)",
            desc="Evaluates sequential (emotion, emotion_mlp) vs joint bottleneck (emotion_joint) on Qwen3-VL 8B under Anchor C.",
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
            id=6,
            codename="6_joint-clip-ft",
            folder="6_joint-clip-ft",
            title="Sequential vs Joint Bottleneck on CLIP-ft Score (Anchor C)",
            desc="Evaluates sequential vs joint bottleneck on fine-tuned CLIP (score) under Anchor C.",
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
            id=7,
            codename="7_joint-clip-ft-emo",
            folder="7_joint-clip-ft-emo",
            title="Sequential vs Joint Bottleneck on CLIP-ft Emotion (Anchor C)",
            desc="Evaluates sequential vs joint bottleneck on fine-tuned CLIP (emotion) under Anchor C.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "clip_ft_emo",
                "--mediators", "emotion,emotion_mlp,emotion_joint",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "C"
            ]
        ),
        SuiteStep(
            id=8,
            codename="8_joint-qwen4b",
            folder="8_joint-qwen4b",
            title="Sequential vs Joint Bottleneck on Qwen3-VL 4B (Anchor C)",
            desc="Evaluates sequential vs joint bottleneck on Qwen3-VL 4B (Layer 15) under Anchor C.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "qwen4b",
                "--mediators", "emotion,emotion_mlp,emotion_joint",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "C"
            ]
        ),

        # --- Group 3: Stage-1 Distributional Mediators Sweeps (Anchor C) ---
        SuiteStep(
            id=9,
            codename="9_dist-clip",
            folder="9_dist-clip",
            title="Distributional Mediators on CLIP Frozen (Anchor C)",
            desc="Evaluates emotion vs emotion_sd vs emotion_hist on Frozen OpenAI CLIP ViT-B/32 under Anchor C.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "clip",
                "--mediators", "emotion,emotion_sd,emotion_hist",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "C"
            ]
        ),
        SuiteStep(
            id=10,
            codename="10_dist-qwen8b",
            folder="10_dist-qwen8b",
            title="Distributional Mediators on Qwen3-VL 8B (Anchor C)",
            desc="Evaluates emotion vs emotion_sd vs emotion_hist on Qwen3-VL 8B under Anchor C.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "qwen8b",
                "--mediators", "emotion,emotion_sd,emotion_hist",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "C"
            ]
        ),
        SuiteStep(
            id=11,
            codename="11_dist-clip-ft",
            folder="11_dist-clip-ft",
            title="Distributional Mediators on CLIP-ft Score (Anchor C)",
            desc="Evaluates emotion vs emotion_sd vs emotion_hist on fine-tuned CLIP (score) under Anchor C.",
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
            id=12,
            codename="12_dist-clip-ft-emo",
            folder="12_dist-clip-ft-emo",
            title="Distributional Mediators on CLIP-ft Emotion (Anchor C)",
            desc="Evaluates emotion vs emotion_sd vs emotion_hist on fine-tuned CLIP (emotion) under Anchor C.",
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
            id=13,
            codename="13_dist-qwen4b",
            folder="13_dist-qwen4b",
            title="Distributional Mediators on Qwen3-VL 4B (Anchor C)",
            desc="Evaluates emotion vs emotion_sd vs emotion_hist on Qwen3-VL 4B (Layer 15) under Anchor C.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "qwen4b",
                "--mediators", "emotion,emotion_sd,emotion_hist",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "C"
            ]
        ),
    ]
)
