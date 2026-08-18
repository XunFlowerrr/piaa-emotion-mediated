# Duplicate clip_ft_emo efficiency run

`clip_ft_emo_second_run/` is a second execution of the same four commands
that produced `output/efficiency/clip_ft_emo/`. It is kept rather than
deleted because the two runs are **not** bit-identical.

They agree on every reported mean to 4 decimal places:

| mediator   | second run | kept run |
|------------|-----------|----------|
| emotion    | .4131     | .4131    | (variant C)
| identity   | .4288     | .4288    |
| population | .4074     | .4074    |

but 9-72 of 18,576 per-unit rows differ per variant. The differing rows are
whole (fold, domain, n_train, seed, mediator) cells, and their `eff_dof`
differs too -- e.g. fold 2 / fashion / n=50 / seed 0 has dof 6.1 in one run
and 5.1 in the other. That is one step on the ridge alpha grid: two adjacent
alphas scored within the tie tolerance on the validation group, and
platform-level floating point picked a different winner.

So the pipeline is reproducible on one machine but not across machines, and
mean SROCC should be treated as stable to about +/-0.001. Nothing in the
paper should rest on a difference smaller than that.
