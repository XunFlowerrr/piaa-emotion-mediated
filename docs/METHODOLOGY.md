# Methodology

This document explains every decision that affects the numbers reported in
the paper, with the reasoning behind it. Anyone verifying this work should
read this before reading the code.

## 1. Model structure

Every experiment is a two-axis table.

- **Axis one - the mediator**, 7-dimensional, shared across users. It is
  fit only on images seen by the training-group users, then frozen.
- **Axis two - the head**, a single per-user layer. It is fit only on that
  user's own ratings.

Training is **sequential** in both the ridge and MLP cases: the mediator is
fit and frozen first, then the head is fit on the mediator's output. It is
not trained end-to-end.

Mediators compared:

| name | what the mediator is | question it answers |
|---|---|---|
| identity (Direct) | no mediator, raw features | does routing through a mediator help at all |
| emotion (ours) | predicts 7 emotions | does a *meaningful* mediator help |
| pca | unsupervised 7-dim compression | is the gain just dimensionality reduction |
| random | random linear projection to 7 dims | does any 7-dim mediator work |
| shuffled | emotion predictions shuffled across images | does the mediator need to match the image |

## 2. Leak-free data split

The 129 users are partitioned into 10 groups. Each fold uses 7 groups as
train, 1 as validation, and 2 as test, rotating over 5 folds so every user
is a test user exactly once, giving 387 evaluation units (129 users x 3
domains).

This closes three leakage points at once:

1. The mediator is fit only on train-group images, so it never sees a test
   user's images.
2. Every **shared** hyperparameter is selected on the validation group,
   which is disjoint from both train and test users (and whose images are
   disjoint from theirs). See "Where each hyperparameter comes from" below.
3. Each test user's own images are split into support and eval sets, so
   the head never trains on what it's scored on.

### Where each hyperparameter comes from

One rule, applied everywhere:

| component | shared or personal? | hyperparameter selected on |
|---|---|---|
| Stage-1 emotion mediator (ridge) | shared | validation user group |
| Shuffled mediator (ridge) | shared | validation user group |
| Stage-1 emotion mediator (MLP) | shared | validation user group |
| Population / GIAA head (ridge, MLP) | shared | validation user group |
| Pop-zero formula in `efficiency` | shared | validation user group |
| Per-user head (ridge) | personal | that user's own support set (`RidgeCV`) |
| Per-user head (MLP) | personal | 80/20 split of that user's own support set |

The two personal rows cannot use the validation group: those users are
*different people*, and the whole point of a personal head is that it is
fit to one individual's taste. What matters for them is that the user's
50 evaluation images are held out before anything is fit and are never
touched during selection - which `verify --splits` and the fixed
support/eval split below both enforce.

PCA (`n_components=7`) and the random projection have no hyperparameter to
select, so nothing is chosen for them from any data.

`uv run main.py verify --splits` checks points 1 and 3 automatically.

### Support/eval split per user

A user's images in each domain are shuffled with `RandomState(42 +
user_id)`, and **the first 50 images are held out as a fixed eval set**;
the rest form the support pool. Holding out eval first makes results
comparable across rating budgets, because a model trained on 10 images and
one trained on 100 are scored on the exact same images.

## 3. Using only first-session ratings

About 5% of the data (4,509 pairs) are images a user rated twice, in
separate sittings. All experiments **use only the first rating** and
reserve the second one exclusively for measuring test-retest reliability.
This leaves 83,327 of 87,836 rows in use.

"first" means first row in the file. Checked that
both ratings of a pair always land in the same split group.

## 4. Metrics

We report **SROCC and PLCC**, following standard practice in image
aesthetic assessment.

Comparisons between models use a **paired Wilcoxon signed-rank test** on
the same 387 units, since every model is evaluated on the same users and
images.

Tables show mean +/- sd across the 387 units, plus a significance flag from
the paired test, since it's testing per-unit differences, not the intervals.

## 5. Heads

### Ridge

Alpha comes from the same grid everywhere: **11 values from 1e-2 to 1e3**
(`numpy.logspace(-2, 3, 11)`). Features are always standardized first.

How the winner is picked depends on whether the component is shared or
personal (Sec. 2). A **shared** ridge is fit on the train group at each
alpha and scored by MSE on the validation group; the best alpha wins. A
**personal** ridge uses `RidgeCV`'s generalized (efficient leave-one-out)
cross-validation inside that user's own support set.

Fitting a mediator on shuffled labels and then selecting its alpha honestly
drives it to the top of the grid (1e3, i.e. maximal shrinkage), because
there is genuinely no signal for the validation group to reward. That is
the control behaving correctly, not a bug.

We also report effective degrees of freedom, defined as
`tr(Z(Z'Z + alpha*I)^-1 Z')` at the selected alpha, computed on standardized
features and averaged across units. This is defined only for linear heads.

### MLP

A single hidden layer of 128 units, ReLU, trained with MSE loss and no
weight decay (alpha = 0). The learning rate is chosen from 5 values
between 1e-4 and 1e-2, then the model is refit on the full data with the
chosen rate. Which data scores the 5 candidates follows the same rule as
ridge: a shared MLP is scored on the validation user group, a personal MLP
on a 20% split of that user's own support set.

Early stopping is used (validation_fraction 0.15, n_iter_no_change 20, max_iter 2000), which is a stopping rule.
Loss curves in `output/mlp_diagnostics/` show it's actually converging properly, not just cut off early.

## 6. Backbones

| name | feature file | dim |
|---|---|---|
| CLIP frozen | `clip_features.npz` | 512 |
| CLIP-ft (score) | `clip_ftpf_overall_v4_fold{k}.npz` | 768 |
| Qwen3-VL 4B | `vlm4b_LT15.npz` | 2560 |
| Qwen3-VL 8B | `vlm_LT15.npz` | 4096 |

A fine-tuned backbone must always use the **per-fold** version, fine-tuned
only on that fold's train users.

## 7. Two upper bounds

- **GT emotions** uses the user's true emotion ratings instead of the
  mediator's prediction. This is the ceiling of the mediator pathway - how
  well the model would do if the mediator were perfect. It is measured
  within the same session, so it is optimistic.
- **Test-retest reliability** is a user's agreement with their own rating
  across sessions - a more realistic ceiling for deployment.

Neither is included when identifying the best-performing predictive model.

## 8. Faithfulness tests

- **Ablation** holds one emotion constant at its support-set mean (the
  highest-weighted, an average-weighted, or the lowest-weighted concept)
  and re-scores the user, to see how much each concept's presence is
  actually load-bearing for that user's predictions.
- **Formula swap** compares a user's own formula against the
  population-mean formula and against 5 randomly sampled other users'
  formulas, on the same predictions.
- **Weight vs. empirical correlation** compares the 7 weights the ridge
  head learned against the true correlation between each emotion and that
  user's own ratings.

## 9. Reproducibility

Every source of randomness is seeded, so a single run (seed 0) is fully
deterministic. Table 1 is the only experiment whose reported rows include
both a stochastic mediator (random/shuffled) and an MLP head; every other
experiment (`backbone`, `efficiency`, `faithfulness`,
`stage1_emotion_acc`, `stage2_emotion_importance`) only ever reports the
`emotion` mediator with a ridge head, which has no randomness at all -
one run is already the final number for those.

For Table 1, every stochastic point is repeated under **3 seeds (0, 1,
2)** and averaged per unit before summarizing, since a single unlucky
draw for the random/shuffled mediators or the MLP init shouldn't set the
reported number. seed 0 reproduces the original single-seed numbers
bit-for-bit; seeds 1 and 2 offset every base seed below by
`+ run_seed * 1_000_003`.

### What "reproducible" does and does not promise

**Same machine, same environment: byte-identical.** `verify --repro` checks
this and passes on `efficiency`, `stage1_emotion_acc` and
`stage2_emotion_importance`.

**Different platform: identical to about 1e-5, not bit-identical.** A run on
Linux/Python 3.14 against Windows/Python 3.13 agrees to ~1e-5 on SROCC and
~1e-8 on PLCC, because a different BLAS build sums in a different order.
Every number the paper reports to three decimals is unaffected.

The one quantity that was *not* robust to this was the percentage of positive
coefficients, because it is a sign test and ridge shrinks weak emotions to
exactly zero - the sign of a zero is decided by the last bit. That is now
counted with a tolerance (`SIGN_TOL` in `stage2_emotion_importance.py`), so it
agrees across platforms too. Anyone reproducing on other hardware should
expect the third decimal of a correlation to move and nothing else.

| point | base seed source (seed 0) |
|---|---|
| support/eval split per user | `42 + user_id` (never reseeded - fixed across runs) |
| mediator random projection and shuffle | one generator per fold, drawing R first, then the permutation |
| per-user MLP | `user_id` |
| population-level MLP | `100 + fold` |
| MLP emotion mediator | `fold` |
| PCA | 0 (never reseeded - PCA is deterministic given the data) |

The order in which the random and shuffled mediators draw random numbers is
fixed; changing that order changes the numbers even though the method is
unchanged.
