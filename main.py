"""Single entry point for the project.

Examples:
    uv run main.py verify --splits
    uv run main.py table1
    uv run main.py backbone --backbones clip,clip_ft,qwen4b,qwen8b
    uv run main.py efficiency --n-train 10,25,50,100
    uv run main.py faithfulness
    uv run main.py mlp_diagnostics

Every command writes to output/<experiment name>/, always with a config.json next to it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import Config                                    # noqa: E402
from src.data.data import XpassDataset                           # noqa: E402
from src.data.splits import V4Split                              # noqa: E402
from src.modeling.backbones import get_backbone                  # noqa: E402
from src.modeling.pipeline import Pipeline                       # noqa: E402


def build(cfg: Config, backbone_name: str | None = None):
    """Wire up dataset + backbone + split + pipeline from config."""
    ds = XpassDataset(cfg.data_dir, first_session_only=cfg.first_session_only)
    bb = get_backbone(backbone_name or cfg.backbone, cfg.features_dir)
    sp = V4Split(cfg.split_dir, n_folds=cfg.n_folds)
    return ds, bb, sp, Pipeline(cfg, ds, bb, sp)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="main.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment",
                    choices=["table1", "backbone", "efficiency", "faithfulness",
                            "mlp_diagnostics", "stage1_emotion_acc",
                            "stage2_emotion_importance", "fig_faithfulness",
                            "fig_efficiency", "verify"])
    ap.add_argument("--backbone", default="clip", help="default: clip")
    ap.add_argument("--backbones", default="clip,clip_ft,qwen4b,qwen8b")
    ap.add_argument("--n-train", default="10,25,50,100")
    ap.add_argument("--all-sessions", action="store_true",
                    help="disable the first-session filter (not what the paper uses)")
    ap.add_argument("--splits", action="store_true", help="verify: check the data split")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args(argv)

    cfg = Config()
    if args.all_sessions:
        cfg.first_session_only = False
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    cfg.backbone = args.backbone

    if args.experiment == "verify":
        from src.utils import verify
        return 0 if verify.check_splits(cfg) else 1

    ds, bb, sp, pipe = build(cfg, args.backbone)

    # --every experiment--
    if args.experiment == "table1":
        from src.experiments import table1
        table1.run(cfg, pipe, ds)
    elif args.experiment == "backbone":
        from src.experiments import backbone
        backbone.run(cfg, args.backbones.split(","))
    elif args.experiment == "efficiency":
        from src.experiments import efficiency
        efficiency.run(cfg, pipe, [int(x) for x in args.n_train.split(",")])
    elif args.experiment == "faithfulness":
        from src.experiments import faithfulness
        faithfulness.run(cfg, pipe)
    elif args.experiment == "mlp_diagnostics":
        from src.experiments import mlp_diagnostics
        mlp_diagnostics.run(cfg, pipe)
    elif args.experiment == "stage1_emotion_acc":
        from src.experiments import stage1_emotion_acc
        stage1_emotion_acc.run(cfg, pipe)
    elif args.experiment == "stage2_emotion_importance":
        from src.experiments import stage2_emotion_importance
        stage2_emotion_importance.run(cfg, pipe)
    elif args.experiment == "fig_faithfulness":
        from src.utils import fig_faithfulness
        fig_faithfulness.run(cfg, "sd")
        fig_faithfulness.run(cfg, "sem")
    elif args.experiment == "fig_efficiency":
        from src.utils import fig_efficiency
        fig_efficiency.run(cfg, "sd")
        fig_efficiency.run(cfg, "sem")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
