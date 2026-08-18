"""Suite 2: 'hayashi' (Headline runs requested by Hayashi on CLIP post-pull)."""
import sys
from src.utils.suite_engine import Suite, SuiteStep

SUITE = Suite(
    name="hayashi",
    title="Hayashi Headline Exploration Suite",
    desc="Explores joint bottleneck, support-size sweep, MLP heads, and distributional Stage-1 under Anchor C.",
    steps=[
        SuiteStep(
            id=1,
            codename="joint-n100-c",
            title="A.1 Headline Joint vs Seq (n=100, Anchor C)",
            desc="Evaluates sequential vs joint bottleneck on n=100 with Anchor C.",
            cmd=[sys.executable, "main.py", "efficiency", "--backbone", "clip", "--mediators", "emotion,emotion_mlp,emotion_joint", "--n-train", "100", "--seed", "0,1,2", "--stage2", "C"]
        ),
        SuiteStep(
            id=2,
            codename="joint-n100-plain",
            title="A.2 Headline Joint vs Seq (n=100, Plain)",
            desc="Baseline sequential vs joint bottleneck on n=100 without anchoring.",
            cmd=[sys.executable, "main.py", "efficiency", "--backbone", "clip", "--mediators", "emotion,emotion_mlp,emotion_joint", "--n-train", "100", "--seed", "0,1,2", "--stage2", "plain"]
        ),
        SuiteStep(
            id=3,
            codename="joint-sweep-c",
            title="B. Support-size Sweep for Fig 2 (Anchor C)",
            desc="Full support-size sweep (n=10,25,50,100) for sequential vs joint bottleneck.",
            cmd=[sys.executable, "main.py", "efficiency", "--backbone", "clip", "--mediators", "emotion,emotion_mlp,emotion_joint", "--n-train", "10,25,50,100", "--seed", "0,1,2", "--stage2", "C"]
        ),
        SuiteStep(
            id=4,
            codename="mlp-sweep-c",
            title="C. MLP & Ridge Heads under Anchor C",
            desc="Re-evaluates MLP personal heads under residual Anchor C across support sizes.",
            cmd=[sys.executable, "main.py", "efficiency", "--backbone", "clip", "--heads", "ridge,mlp", "--n-train", "10,25,50,100", "--seed", "0,1,2", "--stage2", "C"]
        ),
        SuiteStep(
            id=5,
            codename="dist-sweep-c",
            title="D. Distributional Stage-1 under Anchor C",
            desc="Evaluates distribution-valued Stage-1 (mean, sd, hist) under Anchor C.",
            cmd=[sys.executable, "main.py", "efficiency", "--backbone", "clip", "--mediators", "emotion,emotion_sd,emotion_hist", "--n-train", "10,25,50,100", "--seed", "0,1,2", "--stage2", "C"]
        ),
        SuiteStep(
            id=6,
            codename="table1-mlp-c",
            title="E. Table 1 MLP Grid under Anchor C",
            desc="Recomputes Table 1 MLP rows across all mediators under Anchor C (seeds 0,1,2).",
            cmd=[sys.executable, "main.py", "table1", "--backbone", "clip", "--heads", "mlp", "--stage2", "C"]
        ),
    ]
)
