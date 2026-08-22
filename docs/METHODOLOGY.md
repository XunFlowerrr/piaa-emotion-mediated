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

**Nothing is selected on data the model being scored has seen, and nothing
is selected on a test user.** The validation user group - disjoint from both
the train and the test group - carries every choice. How it is used depends
on whether the component is shared across users or personal to one:

| component | shared or personal? | selected on | criterion |
|---|---|---|---|
| Stage-1 mediators, ridge (`emotion`, `shuffled`, `emotion_sd`, `emotion_hist`, `shuffled35`) | shared | validation user group | MSE over validation images |
| Stage-1 mediator, sequential MLP (`emotion_mlp`) | shared | validation user group | MSE over validation images |
| Stage-1 mediator, joint MLP (`emotion_joint`) | shared | validation user group | its own joint loss |
| Population / GIAA head (ridge, MLP) | shared | validation user group | MSE over validation images |
| Variant B/C population anchor | shared | validation user group | MSE over validation images |
| **Per-user head (ridge, lasso, elastic, MLP)** | personal | **validation user group, by mirroring the test protocol** | mean SROCC over validation user-units |

#### The personal head: mirroring the test protocol

A personal head is fit to one person from a few dozen ratings. Choosing its
hyperparameter on the pooled training group - which is what `RidgeCV`'s
internal generalized cross-validation would do - tunes it for a regime the
head never faces. So the test protocol is reproduced inside the validation
group instead (`Pipeline.select_personal_hyperparam`):

1. For every validation user and domain, run **the identical support/eval
   split used for test users**: shuffle that user's images with
   `RandomState(42 + user_id)`, hold out the first 50 as a fixed inner
   evaluation set, draw the support set from the remainder. This is literally
   the same function (`Pipeline.iter_units`) with `users=fold.val_users`; no
   split logic differs between the two groups.
2. For each candidate value, fit a personal head on each validation user's
   support set, score it with SROCC on that user's own inner evaluation set,
   and average across all validation user-domain units.
3. Freeze the best value and apply that single fixed value to every test
   user's personal head. No test-group data is touched during selection, and
   no test user's head is chosen by their own held-out performance.

Selection runs **separately per (fold, domain, support size, mediator,
head)**. Per support size because the optimal penalty depends strongly on it
- 10 ratings need far heavier regularization than 100. Per domain is a
choice, not a necessity; we state it here because either is defensible.

The same procedure carries **every** condition - Direct, Random, Shuffled,
PCA, Hybrid - so a difference between rows is a difference between mediators
and not between how carefully each was tuned.

Two components are deliberately outside it, and both are stated rather than
hidden:

- The **GIAA / population head** has no support set and no per-user fit, so
  the mirrored per-user protocol cannot be applied to it. Its hyperparameter
  is still chosen on the validation group, scored by MSE over that group's
  images.
- The **`gt_emotion` upper bound** keeps its own per-user selection
  (`RidgeCV` inside the user's support set). It is an oracle ceiling that
  uses true emotion ratings, reported for reference and excluded from every
  fairness comparison in the paper.

PCA (`n_components=7`), the random projection and Direct have no Stage-1
hyperparameter, so nothing is chosen for them from any data. Their Stage-2
personal heads still go through the protocol above.

`uv run main.py verify --splits` checks the split itself; the
`selection*.csv` described next records what every one of these choices
came out as.

### What each run selected: `selection*.csv`

Every run writes a selection log next to its results (`selection<tag>_seed<s>.csv`
in `output/table1/`, `selection<tag>.csv` in `output/efficiency/<backbone>/` and
`output/backbone/`). Without it a finished run cannot answer "which learning
rate did the MLP head get?" - the winner was passed to a constructor and
dropped, and recovering it meant re-running the selection, which costs as much
as the run.

One row per **candidate**, not per winner, because the winner alone hides the
two things that decide whether a selection meant anything:

| column | what it tells you |
|---|---|
| `selected`, `rank` | which candidate won, and where the rest landed |
| `score`, `criterion` | the value it was ranked by (`val_mse`, `val_srocc_mean`, `val_joint_loss`) |
| `margin` | how far ahead of the best non-tied candidate it was. Near zero means the choice is noise and the next seed will overturn it |
| `edge` | which axis ran out of grid, and at which end. `alpha:max` means the strongest weight decay on offer won, so the number is where the grid stopped, not where the optimum is |
| `tie_size` | candidates the selector treated as tied (see `ALPHA_TIE_RTOL`) |
| `n_val_scored`, `n_val_kind`, `n_val_users` | how much held-out data the choice rests on. The personal head averages SROCC over validation *user-units*, and a fold has as few as 6 validation users |

`kind` separates ranked candidates (`selection`) from measured facts about the
model that won (`diagnostic`): epochs actually run (`mean_n_iter`,
`frac_hit_max_iter` - `tol=0` means an MLP runs its full budget rather than
converging, and that is worth having next to the score), support size,
mediator width, and mean effective d.o.f.

`selection_log.summarize(records)` collapses it to one row per selection.

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

Alpha comes from the same grid everywhere: **17 values from 1e-2 to 1e6**
(`numpy.logspace(-2, 6, 17)`). Features are always standardized first. The
grid runs well past the point where a 7- or 512-feature head on <=100
standardized samples is fully shrunk, so the top of it is a floor the
selector can actually reach rather than a cliff it is cut off before -
which matters most under variants B and C, where full shrinkage lands on
the population formula instead of on a constant.

How the winner is picked depends on whether the component is shared or
personal (Sec. 2). A **shared** ridge is fit on the train group at each
alpha and scored by MSE on the validation group. A **personal** ridge is
selected by mirroring the test protocol inside the validation group and
averaging SROCC over validation user-units; the value is then frozen and
applied to every test user.

Ties are broken toward the **strongest** penalty - among alphas that are
indistinguishable on the validation group, the most regularized one is the
conservative choice, and picking it by rule keeps the result from depending
on the last few floating-point bits.

Fitting a mediator on shuffled labels and then selecting its alpha honestly
drives it to the top of the grid (1e3, i.e. maximal shrinkage), because
there is genuinely no signal for the validation group to reward. That is
the control behaving correctly, not a bug.

We also report effective degrees of freedom, defined as
`tr(Z(Z'Z + alpha*I)^-1 Z')` at the selected alpha, computed on standardized
features and averaged across units. This is defined only for linear heads.

### MLP

**A single hidden layer of 128 units, ReLU, used throughout** - the same
width in the extractor (features -> 7 concepts) and in the predictor
(7 concepts -> 1 score). Trained with MSE loss, Adam, and **no weight decay
(alpha = 0, fixed in advance, not tuned)**.

**No early stopping, and a fixed epoch budget of 500.** With support sets as
small as 10 ratings, sklearn's internal 85/15 validation split would leave
one or two samples and give no usable stopping signal, so `early_stopping`
is off. That alone is not enough: sklearn also halts on a training-loss
plateau, so `tol=0` and `n_iter_no_change=max_iter` are set at construction
as well. Every MLP therefore really does spend the same 500 epochs, which
the selection log records per run (`mean_n_iter`, `frac_hit_max_iter`) rather
than leaving it to be assumed.

The epoch count is stated in advance rather than tuned. It follows that the
MLPs are not run to convergence; they are run to a fixed, equal budget, which
is what makes the rows comparable.

**The learning rate is the only thing selected**, from five values
(`1e-4, 3e-4, 1e-3, 3e-3, 1e-2`), by the same procedure as every other
hyperparameter (Sec. 2): a shared MLP is scored by MSE on the validation user
group, a personal MLP by mean SROCC over validation user-units under the
mirrored test protocol. Ties break toward the **smallest** step - the
conservative direction for a step size, opposite to ridge's tie-break toward
the strongest penalty.

Everything above is built in one place (`make_mlp` in
`src/modeling/heads.py`), so the extractor and the predictor cannot drift
apart. `MLPHead.fit` raises rather than selecting a rate itself when none is
frozen, so a fallback to the user's own support set cannot come back by
accident.

#### The three MLP rows

| reported as | Stage-1 (extractor) | Stage-2 (predictor) | mediator / head |
|---|---|---|---|
| Ridge -> Ridge | ridge d->7 | ridge 7->1 | `emotion` / `ridge` |
| MLP -> MLP sequential | MLP d->128->7, emotion loss only | MLP 7->128->1 | `emotion_mlp` / `mlp` |
| MLP -> MLP joint | MLP d->128->7->1, emotion + score loss | MLP 7->128->1 | `emotion_joint` / `mlp` |

The joint extractor's score head reads the **seven concepts**, not the hidden
layer, so the score gradient is forced through the bottleneck - hanging it off
the hidden layer instead would leave the concepts under emotion supervision
alone, which is sequential training wearing a joint label. Its loss is
`MSE(concepts) + w * MSE(score)` with `w = 1`, fixed in advance; the two
targets sit on comparable scales (emotion sd ~ 0.55, score sd ~ 0.70).

The mixed combinations the grid also produces (`emotion_mlp` + `ridge`, and
so on) are not reported.

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
