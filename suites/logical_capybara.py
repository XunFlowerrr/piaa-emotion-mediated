"""Suite 11: 'logical_capybara'.

Table 1 Control Baselines Suite (identity, random, pca under MLP Head, Stage-2 C, Seeds 0, 1, 2).
"""
import sys
from src.utils.suite_engine import Suite, SuiteStep

SUITE = Suite(
    name="logical_capybara",
    title="Table 1 Control Baselines Suite (Stage-2 C, MLP Head, 3 Seeds)",
    desc="Runs control baselines (identity, random, pca) under MLP personal head and Anchor C across 3 seeds.",
    steps=[
        SuiteStep(
            id=1,
            codename="1_ctrl-mlp-s0",
            folder="1_ctrl-mlp-s0",
            title="MLP Head Controls (identity, random, pca) - Seed 0",
            desc="Table 1 controls under MLP head on Qwen8B at n=100 ratings/user, seed 0, Anchor C.",
            cmd=[
                sys.executable, "main.py", "table1",
                "--stage2", "C",
                "--heads", "mlp",
                "--mediators", "identity,random,pca",
                "--seed", "0",
            ]
        ),
        SuiteStep(
            id=2,
            codename="2_ctrl-mlp-s1",
            folder="2_ctrl-mlp-s1",
            title="MLP Head Controls (identity, random, pca) - Seed 1",
            desc="Table 1 controls under MLP head on Qwen8B at n=100 ratings/user, seed 1, Anchor C.",
            cmd=[
                sys.executable, "main.py", "table1",
                "--stage2", "C",
                "--heads", "mlp",
                "--mediators", "identity,random,pca",
                "--seed", "1",
            ]
        ),
        SuiteStep(
            id=3,
            codename="3_ctrl-mlp-s2",
            folder="3_ctrl-mlp-s2",
            title="MLP Head Controls (identity, random, pca) - Seed 2",
            desc="Table 1 controls under MLP head on Qwen8B at n=100 ratings/user, seed 2, Anchor C.",
            cmd=[
                sys.executable, "main.py", "table1",
                "--stage2", "C",
                "--heads", "mlp",
                "--mediators", "identity,random,pca",
                "--seed", "2",
            ]
        ),
    ]
)
