# output/efficiency

One folder per backbone. Two files per run:

- `raw{tag}.csv`      one row per (user-domain unit, seed) -- everything else rebuilds from this
- `summary{tag}.csv`  what gets reported

`{tag}` encodes anything that changes the numbers: variant, heads, backbone,
mediator set. See `src/experiments/efficiency.py::_tag`.

## Cross-machine reproducibility

clip_ft_emo was run twice, once here and once on a MacBook. Every reported
mean agreed to 4 decimal places, but 9-72 of 18,576 per-unit rows differed
per variant. The differing rows are whole (fold, domain, n_train, seed,
mediator) cells whose frozen ridge alpha landed one grid step apart --
e.g. fold 2 / fashion / n=50 / seed 0 had effective dof 6.1 in one run and
5.1 in the other. Two adjacent alphas had tied on the validation group and
platform-level floating point picked different winners.

The second run was deleted after this check: it was not better, it was the
same experiment. Treat mean SROCC as stable to about +/-0.001 and do not
rest a conclusion on a smaller difference.
