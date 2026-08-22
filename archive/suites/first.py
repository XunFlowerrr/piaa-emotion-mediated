"""Suite 1: 'first' (clip_ft_emo 4-variant efficiency sweep)."""
import sys
from src.utils.suite_engine import Suite, SuiteStep

SUITE = Suite(
    name="first",
    title="CLIP-FT-Emo Baseline 4-Variant Sweep",
    desc="Initial baseline evaluation on emotion fine-tuned CLIP across 4 Stage-2 variants (plain, A, B, C).",
    steps=[
        SuiteStep(
            id=1,
            codename="ft-plain",
            title="Variant Plain (Ordinary Ridge)",
            desc="Fits personal ridge on clip_ft_emo features shrinking toward 0.",
            cmd=[sys.executable, "main.py", "efficiency", "--backbone", "clip_ft_emo", "--n-train", "10,25,50,100", "--seed", "0,1,2", "--stage2", "plain"]
        ),
        SuiteStep(
            id=2,
            codename="ft-anchor-a",
            title="Variant A (GIAA Feature)",
            desc="Appends GIAA population score as an extra feature.",
            cmd=[sys.executable, "main.py", "efficiency", "--backbone", "clip_ft_emo", "--n-train", "10,25,50,100", "--seed", "0,1,2", "--stage2", "A"]
        ),
        SuiteStep(
            id=3,
            codename="ft-anchor-b",
            title="Variant B (Weight Shrinkage)",
            desc="Shrinks personal ridge weights toward pooled population model weights.",
            cmd=[sys.executable, "main.py", "efficiency", "--backbone", "clip_ft_emo", "--n-train", "10,25,50,100", "--seed", "0,1,2", "--stage2", "B"]
        ),
        SuiteStep(
            id=4,
            codename="ft-anchor-c",
            title="Variant C (Residual Fit)",
            desc="Fits personal ridge on residuals against GIAA population prediction.",
            cmd=[sys.executable, "main.py", "efficiency", "--backbone", "clip_ft_emo", "--n-train", "10,25,50,100", "--seed", "0,1,2", "--stage2", "C"]
        ),
    ]
)
