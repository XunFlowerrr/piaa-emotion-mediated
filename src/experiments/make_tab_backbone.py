r"""Render output/backbone/summary.csv as the Table 2 LaTeX body.

Reads only the summary written by `main.py backbone` -- every number in the
table comes from that file, nothing is recomputed here, so the table cannot
drift from the run that produced it.

Conventions kept from the previous version of the table:
  bold      the better mean of Direct / Hybrid, per cell
  \dagger   on the Direct cell when Hybrid vs. Direct is significant
            (paired Wilcoxon, p < 0.05) for that domain and metric
  x_{\pm s} mean and SEM, both printed without the leading zero

Usage:  python -m src.experiments.make_tab_backbone [out.tex]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:                        # the repo path has Thai characters in it
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# (summary.csv column prefix, printed heading)
DOMAIN_COLS = [("art_", "Artwork"), ("fashion_", "Fashion"),
               ("landscape_", "Landscape"), ("", "Average")]


def _num(x: float) -> str:
    """.478 -- three decimals, no leading zero (negatives keep the sign)."""
    s = f"{x:.3f}"
    return s.replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s


def _cell(mean: float, sem: float, bold: bool, dagger: bool) -> str:
    body = f"{_num(mean)}_{{\\pm {_num(sem)}}}"
    if bold:
        body = r"\bm{" + body + "}"
    if dagger:
        body += r"^\dagger"
    return f"${body}$"


def build(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{ll cc cc cc cc}",
        r"\toprule",
        "",
        r"\multirow{2}{*}{Backbone} & \multirow{2}{*}{Mediator} & ",
        r"\multicolumn{2}{c}{Artwork} & \multicolumn{2}{c}{Fashion} &",
        r"\multicolumn{2}{c}{Landscape} & \multicolumn{2}{c}{Average} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}",
        r" & & SROCC & PLCC & SROCC & PLCC & SROCC & PLCC & SROCC & PLCC \\",
        r"\midrule",
    ]

    for i, row in enumerate(summary.itertuples()):
        if i:
            lines.append(r"\addlinespace")
        d = row._asdict()
        cells = {"Direct": [], "Hybrid": []}
        for pre, _ in DOMAIN_COLS:
            for m in ("srocc", "plcc"):
                hyb_best = bool(d[f"{pre}hybrid_{m}_best"])
                sig = bool(d[f"{pre}hybrid_{m}_sig"])
                cells["Direct"].append(_cell(
                    d[f"{pre}direct_{m}_mean"], d[f"{pre}direct_{m}_sem"],
                    bold=not hyb_best, dagger=sig))
                cells["Hybrid"].append(_cell(
                    d[f"{pre}hybrid_{m}_mean"], d[f"{pre}hybrid_{m}_sem"],
                    bold=hyb_best, dagger=False))
        lines.append(r"\multirow{2}{*}{" + str(d["label"]) + "} ")
        for name in ("Direct", "Hybrid"):
            lines.append(f" & {name} & " + " & ".join(cells[name]) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = Path(__file__).resolve().parents[2]
    src = root / "output" / "backbone" / "summary.csv"
    if not src.exists():
        raise SystemExit(f"missing {src} -- run: main.py backbone")

    summary = pd.read_csv(src)

    # the table is only a paired comparison if every backbone kept the same
    # units; refuse to render a table that quietly is not
    for m in ("srocc", "plcc"):
        for pre, name in DOMAIN_COLS:
            n = summary[f"{pre}n_{m}"].unique()
            if len(n) != 1:
                raise SystemExit(
                    f"unit counts differ across backbones for {name}/{m}: "
                    f"{dict(zip(summary.label, summary[f'{pre}n_{m}']))}")
    print("units per backbone:",
          {name: int(summary[f"{pre}n_srocc"].iloc[0]) for pre, name in DOMAIN_COLS})

    tex = build(summary)
    out = Path(argv[0]) if argv else root / "output" / "backbone" / "tab_backbone.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    print(f"wrote {out}")
    print(tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
