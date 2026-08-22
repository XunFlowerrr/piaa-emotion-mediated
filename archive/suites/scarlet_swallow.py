"""Suite: 'scarlet_swallow'.

Comprehensive Plain (No-Anchor) Sweeps & Qwen3-VL 8B Diagnostics Suite.
10 Flattened Steps:
- Steps 1-5: Plain (No-Anchor) Baseline Sweeps across all 5 backbones (qwen8b, clip, clip_ft, clip_ft_emo, qwen4b)
- Steps 6-7: Plain (No-Anchor) Distributional Sweeps (qwen8b, clip)
- Steps 8-10: Faithfulness, Stage-1 Emotion Accuracy, and Stage-2 Feature Importance on Qwen3-VL 8B
"""
import sys
from src.utils.suite_engine import Suite, SuiteStep

SUITE = Suite(
    name="scarlet_swallow",
    title="Plain Sweeps, Plain Distributions & Qwen8B Diagnostics Suite",
    desc="Comprehensive 10-step sweep under Stage-2 Plain (no anchor) comparing standard baselines (1-5), plain distributions (6-7), and Qwen8B diagnostic investigations (faithfulness, stage1 accuracy, stage2 importance, 8-10).",
    steps=[
        # --- Group 1: Plain Baseline Sweeps (No Anchor) ---
        SuiteStep(
            id=1,
            codename="1_plain-qwen8b",
            folder="1_plain-qwen8b",
            title="Plain Stage-2 Baseline Sweep on Qwen3-VL 8B (No Anchor)",
            desc="Full Stage-2 controls (identity, pca, emotion, random, shuffled) on Qwen3-VL 8B under Stage-2 plain.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "qwen8b",
                "--mediators", "identity,pca,emotion,random,shuffled",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "plain"
            ]
        ),
        SuiteStep(
            id=2,
            codename="2_plain-clip",
            folder="2_plain-clip",
            title="Plain Stage-2 Baseline Sweep on Frozen CLIP (No Anchor)",
            desc="Full Stage-2 controls (identity, pca, emotion, random, shuffled) on Frozen OpenAI CLIP under Stage-2 plain.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "clip",
                "--mediators", "identity,pca,emotion,random,shuffled",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "plain"
            ]
        ),
        SuiteStep(
            id=3,
            codename="3_plain-clip-ft",
            folder="3_plain-clip-ft",
            title="Plain Stage-2 Baseline Sweep on CLIP-ft Score (No Anchor)",
            desc="Full Stage-2 controls on fine-tuned CLIP (overall aesthetic score) under Stage-2 plain.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "clip_ft",
                "--mediators", "identity,pca,emotion,random,shuffled",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "plain"
            ]
        ),
        SuiteStep(
            id=4,
            codename="4_plain-clip-ft-emo",
            folder="4_plain-clip-ft-emo",
            title="Plain Stage-2 Baseline Sweep on CLIP-ft Emotion (No Anchor)",
            desc="Full Stage-2 controls on fine-tuned CLIP (emotion supervision) under Stage-2 plain.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "clip_ft_emo",
                "--mediators", "identity,pca,emotion,random,shuffled",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "plain"
            ]
        ),
        SuiteStep(
            id=5,
            codename="5_plain-qwen4b",
            folder="5_plain-qwen4b",
            title="Plain Stage-2 Baseline Sweep on Qwen3-VL 4B (No Anchor)",
            desc="Full Stage-2 controls on Qwen3-VL 4B (Layer 15) under Stage-2 plain.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "qwen4b",
                "--mediators", "identity,pca,emotion,random,shuffled",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "plain"
            ]
        ),

        # --- Group 2: Plain Distributional Sweeps (No Anchor) ---
        SuiteStep(
            id=6,
            codename="6_dist-plain-qwen8b",
            folder="6_dist-plain-qwen8b",
            title="Plain Distributional Mediators on Qwen3-VL 8B (No Anchor)",
            desc="Evaluates emotion vs emotion_sd vs emotion_hist on Qwen3-VL 8B under Stage-2 plain.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "qwen8b",
                "--mediators", "emotion,emotion_sd,emotion_hist",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "plain"
            ]
        ),
        SuiteStep(
            id=7,
            codename="7_dist-plain-clip",
            folder="7_dist-plain-clip",
            title="Plain Distributional Mediators on Frozen CLIP (No Anchor)",
            desc="Evaluates emotion vs emotion_sd vs emotion_hist on Frozen CLIP under Stage-2 plain.",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "clip",
                "--mediators", "emotion,emotion_sd,emotion_hist",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "plain"
            ]
        ),

        # --- Group 3: Diagnostics on Qwen3-VL 8B ---
        SuiteStep(
            id=8,
            codename="8_faithfulness-qwen8b",
            folder="8_faithfulness-qwen8b",
            title="Faithfulness & Formula Swap Analysis on Qwen3-VL 8B",
            desc="Measures faithfulness, ablation impact, and weight vs empirical correlations on Qwen3-VL 8B.",
            cmd=[
                sys.executable, "main.py", "faithfulness",
                "--backbone", "qwen8b"
            ]
        ),
        SuiteStep(
            id=9,
            codename="9_stage1-acc-qwen8b",
            folder="9_stage1-acc-qwen8b",
            title="Stage-1 Emotion Prediction Accuracy on Qwen3-VL 8B",
            desc="Evaluates population-level Stage-1 regression accuracy for 7 emotion concepts on Qwen3-VL 8B.",
            cmd=[
                sys.executable, "main.py", "stage1_emotion_acc",
                "--backbone", "qwen8b"
            ]
        ),
        SuiteStep(
            id=10,
            codename="10_stage2-imp-qwen8b",
            folder="10_stage2-imp-qwen8b",
            title="Stage-2 Emotion Feature Importance on Qwen3-VL 8B",
            desc="Evaluates personal user regression weights and feature importance on Qwen3-VL 8B.",
            cmd=[
                sys.executable, "main.py", "stage2_emotion_importance",
                "--backbone", "qwen8b"
            ]
        ),
    ]
)
