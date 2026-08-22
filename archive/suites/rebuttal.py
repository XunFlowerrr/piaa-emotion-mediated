"""Suite 3: 'rebuttal' (Full systematic reviewer rebuttal suite)."""
import sys
from src.utils.suite_engine import Suite, SuiteStep

SUITE = Suite(
    name="rebuttal",
    title="Reviewer Rebuttal & Systematic Comparison Suite",
    desc="Systematic comparison across Anchor C and Plain baselines for bottleneck architectures, distributional Stage-1, and MLP heads.",
    steps=[
        SuiteStep(
            id=1,
            codename="joint-c",
            title="Joint vs Sequential Bottleneck (Anchor C)",
            desc="Tests sequential (emotion, emotion_mlp) vs joint (emotion_joint) Stage-1 bottleneck with Anchor C.",
            cmd=[sys.executable, "main.py", "efficiency", "--backbone", "clip", "--mediators", "emotion,emotion_mlp,emotion_joint", "--n-train", "10,25,50,100", "--seed", "0,1,2", "--stage2", "C"]
        ),
        SuiteStep(
            id=2,
            codename="joint-plain",
            title="Joint vs Sequential Bottleneck (Plain / Unanchored)",
            desc="Baseline comparison for sequential vs joint Stage-1 without population anchoring.",
            cmd=[sys.executable, "main.py", "efficiency", "--backbone", "clip", "--mediators", "emotion,emotion_mlp,emotion_joint", "--n-train", "10,25,50,100", "--seed", "0,1,2", "--stage2", "plain"]
        ),
        SuiteStep(
            id=3,
            codename="dist-c",
            title="Distributional Stage-1 (Anchor C)",
            desc="Tests distribution-valued Stage-1 (mean, sd, hist) under Anchor C.",
            cmd=[sys.executable, "main.py", "efficiency", "--backbone", "clip", "--mediators", "emotion,emotion_sd,emotion_hist", "--n-train", "10,25,50,100", "--seed", "0,1,2", "--stage2", "C"]
        ),
        SuiteStep(
            id=4,
            codename="dist-plain",
            title="Distributional Stage-1 (Plain / Unanchored)",
            desc="Baseline comparison for distribution-valued Stage-1 without population anchoring.",
            cmd=[sys.executable, "main.py", "efficiency", "--backbone", "clip", "--mediators", "emotion,emotion_sd,emotion_hist", "--n-train", "10,25,50,100", "--seed", "0,1,2", "--stage2", "plain"]
        ),
        SuiteStep(
            id=5,
            codename="mlp-head-c",
            title="Personal MLP & Ridge Heads (Anchor C)",
            desc="Efficiency sweep comparing Ridge vs MLP personal heads with residual anchoring (Variant C).",
            cmd=[sys.executable, "main.py", "efficiency", "--backbone", "clip", "--heads", "ridge,mlp", "--n-train", "10,25,50,100", "--seed", "0,1,2", "--stage2", "C"]
        ),
        SuiteStep(
            id=6,
            codename="table1-mlp-c",
            title="Table 1 Full Grid for MLP (Anchor C)",
            desc="Recomputes Table 1 MLP rows across all 5 mediators under Anchor C (seeds 0,1,2).",
            cmd=[sys.executable, "main.py", "table1", "--backbone", "clip", "--heads", "mlp", "--stage2", "C"]
        ),
    ]
)
