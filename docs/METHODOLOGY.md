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
2. Hyperparameters are selected on the validation group, which is disjoint
   from test.
3. Each test user's own images are split into support and eval sets.

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

`RidgeCV` selects alpha by generalized cross-validation from a grid of
**11 values from 1e-2 to 1e3** (`numpy.logspace(-2, 3, 11)`). Features are
always standardized first.

We also report effective degrees of freedom, defined as
`tr(Z(Z'Z + alpha*I)^-1 Z')` at the selected alpha, computed on standardized
features and averaged across units. This is defined only for linear heads.

### MLP

A single hidden layer of 128 units, ReLU, trained with MSE loss and no
weight decay (alpha = 0). Learning rate is chosen from 5 values between
1e-4 and 1e-2 by a 20% validation split, then refit on the full data with
the chosen rate.

Early stopping is used (validation_fraction 0.15, n_iter_no_change 20, max_iter 2000), which is a stopping rule.
Loss curves in `output/mlp_diagnostics/` show it's actually converging properly, not just cut off early.

## 6. Backbones

| name | feature file | dim |
|---|---|---|
| CLIP frozen | `clip_features.npz` | 512 |
| CLIP-ft (score) | `clip_ftpf_overall_v4_fold{k}.npz` | 768 |
| Qwen3-VL 4B | `vlm4b_LT17.npz` | 2560 |
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

- **Formula swap** compares a user's own formula against the
  population-mean formula and against 5 randomly sampled other users'
  formulas, on the same predictions.
- **Weight vs. empirical correlation** compares the 7 weights the ridge
  head learned against the true correlation between each emotion and that
  user's own ratings.

## 9. Reproducibility

Every source of randomness is deterministic.

| point | seed source |
|---|---|
| support/eval split per user | `42 + user_id` |
| mediator random projection and shuffle | one generator per fold, drawing R first, then the permutation |
| per-user MLP | `user_id` |
| population-level MLP | `100 + fold` |
| MLP emotion mediator | `fold` |
| PCA | 0 |

The order in which the random and shuffled mediators draw random numbers is
fixed; changing that order changes the numbers even though the method is
unchanged.
