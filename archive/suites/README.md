# Archived experiment suites

Every suite that has been run and reported, retired from `suites/` so
`uv run suite.py list` shows only what is still live. Nothing here is deleted
work -- the results these produced are in `archive-output/<codename>/`, and the
run-by-run record is in `EXPERIMENT_TRACKER.md`.

Each suite is two files kept together, because they are useless apart:

| file | what it is |
|---|---|
| `<codename>.py` | the `SUITE` definition -- steps, CLI commands, output folders |
| `run_<codename>.py` | the standalone runner `suite.py new` generated for it |

## Restoring one

Suites are discovered by globbing `suites/*.py` (see `suites/__init__.py`), so
a suite is live exactly when its definition sits in that directory. Move both
files back:

```bash
mv archive/suites/<codename>.py suites/
mv archive/suites/run_<codename>.py .
uv run suite.py show <codename>
```

The runner has to go back to the project root: it does `from suites.<codename>
import SUITE`, which only resolves from there.

## What stayed behind

`suite.py` and `src/utils/suite_engine.py` are the infrastructure, not the
experiments -- they still build and run new suites. `archive/run_batch.py` and
`archive/run_batch_hayashi.py` are older still: hand-written batch scripts from
before the suite engine existed, which is why they are not in this directory.
