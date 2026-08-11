"""Pipeline -- wires Backbone + Mediator + Head together and runs the v4
evaluation protocol.

The whole thing is a two-axis grid, Mediator x Head. Every row of Table 1
is one cell in that grid, so it's written as a single loop rather than a
separate function per baseline (less duplication, less chance of updating
one baseline and forgetting another).

Per (fold, domain):
  1. pull train-user images -> features Xg, population-mean emotions Eg
  2. fit every mediator on (Xg, Eg), freeze
  3. for each test user: split support/eval, fit the head on support,
     score on the fixed eval set
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.data import CORE7, DOMAINS, XpassDataset
from src.data.splits import V4Split, per_user_split, user_rng
from src.modeling.heads import make_head
from src.modeling.mediators import build_shared_mediators, fit_emotion_mlp
from src.utils.metrics import evaluate


@dataclass
class UserUnit:
    """One evaluation unit = one user x one domain (387 total)."""
    fold: int
    domain: str
    user_id: int
    X_train: np.ndarray      # support image features
    X_eval: np.ndarray       # eval image features
    y_train: np.ndarray      # user's scores on support
    y_eval: np.ndarray
    E_train: np.ndarray      # user's true emotion ratings on support (upper bound)
    E_eval: np.ndarray


class Pipeline:
    def __init__(self, cfg, dataset: XpassDataset, backbone, split: V4Split):
        self.cfg = cfg
        self.ds = dataset
        self.backbone = backbone
        self.split = split

    def iter_units(self, fold, domain: str, feats, n_train: int | None = None):
        """Evaluation units for one fold/domain."""
        cfg = self.cfg
        n_train = n_train or cfg.n_train
        sub = self.ds.subset(domain=domain)
        sub = sub[sub["stimulus_id"].astype(str).isin(feats)]

        for uid in sorted(fold.test_users):
            du = sub[sub["user_id"] == uid]
            stim = du["stimulus_id"].astype(str).unique()
            if len(stim) < n_train + cfg.min_test:
                continue
            rng = user_rng(cfg.split_seed, uid)
            tr_pool, ev_ids = per_user_split(stim, cfg.n_eval, rng)
            agg = self.ds.per_stimulus(du)
            tr = [s for s in tr_pool if s in agg.index][:n_train]
            ev = [s for s in ev_ids if s in agg.index]
            if len(tr) < n_train or len(ev) < cfg.min_test:
                continue
            yield UserUnit(
                fold=fold.index, domain=domain, user_id=int(uid),
                X_train=self.backbone.matrix(feats, tr),
                X_eval=self.backbone.matrix(feats, ev),
                y_train=agg.loc[tr, "overall"].to_numpy(float),
                y_eval=agg.loc[ev, "overall"].to_numpy(float),
                E_train=agg.loc[tr, CORE7].to_numpy(float),
                E_eval=agg.loc[ev, CORE7].to_numpy(float),
            )

    def shared_context(self, fold, domain: str, feats, with_emotion_mlp: bool = False):
        """Population-level data for this fold/domain, plus fitted mediators."""
        gen = self.ds.subset(domain=domain, users=fold.train_users)
        gen = gen[gen["stimulus_id"].astype(str).isin(feats)]
        g = self.ds.per_stimulus(gen)
        Xg = self.backbone.matrix(feats, g.index)
        Eg = g[CORE7].to_numpy(float)
        yg = g["overall"].to_numpy(float)

        emo_mlp = (fit_emotion_mlp(Xg, Eg, self.cfg, seed=fold.index)
                   if with_emotion_mlp else None)
        meds = build_shared_mediators(Xg, Eg, self.cfg, fold.index,
                                      emotion_mlp=emo_mlp)
        return Xg, Eg, yg, meds

    def run_grid(self, mediators: list[str], heads: list[str],
                 n_train: int | None = None, include_population: bool = True,
                 include_gt_upper_bound: bool = True,
                 domains: list[str] | None = None) -> pd.DataFrame:
        """Loop (fold, domain, user) x (mediator, head), return per-unit results.

        include_population      add the no-personalization (GIAA) baseline
        include_gt_upper_bound  add the ceiling that uses true emotion ratings
        """
        cfg = self.cfg
        domains = domains or DOMAINS
        need_mlp_mediator = "mlp" in heads and "emotion" in mediators
        rows = []

        for fold in self.split.folds():
            feats = self.backbone.features_for_fold(fold.index)
            for dom in domains:
                Xg, Eg, yg, meds = self.shared_context(
                    fold, dom, feats, with_emotion_mlp=need_mlp_mediator)

                # baseline that never sees the target user's own ratings (GIAA)
                pop_models = {}
                if include_population:
                    for h in heads:
                        seed = 100 + fold.index if h == "mlp" else 0
                        pop_models[h] = make_head(h, cfg, seed=seed).fit(Xg, yg)

                for unit in self.iter_units(fold, dom, feats, n_train=n_train):
                    base = dict(fold=unit.fold, domain=unit.domain,
                                user_id=unit.user_id)

                    for h in heads:
                        if include_population:
                            p = pop_models[h].predict(unit.X_eval)
                            rows.append({**base, "mediator": "population", "head": h,
                                         "eff_dof": np.nan,
                                         **evaluate(unit.y_eval, p)})

                        for mname in mediators:
                            # with head=mlp, the emotion mediator is also an MLP
                            key = ("emotion_mlp" if (mname == "emotion" and h == "mlp"
                                                     and "emotion_mlp" in meds)
                                   else mname)
                            med = meds[key]
                            M_tr = med.transform(unit.X_train)
                            M_ev = med.transform(unit.X_eval)
                            head = make_head(h, cfg, seed=unit.user_id).fit(M_tr, unit.y_train)
                            p = head.predict(M_ev)
                            rows.append({**base, "mediator": mname, "head": h,
                                         "eff_dof": head.effective_dof(),
                                         **evaluate(unit.y_eval, p)})

                    if include_gt_upper_bound:
                        head = make_head("ridge", cfg).fit(unit.E_train, unit.y_train)
                        p = head.predict(unit.E_eval)
                        rows.append({**base, "mediator": "gt_emotion", "head": "ridge",
                                     "eff_dof": head.effective_dof(),
                                     **evaluate(unit.y_eval, p)})
            print(f"  fold {fold.index} done ({len(rows)} rows)", flush=True)
        return pd.DataFrame(rows)

    def collect_user_heads(self, mediator: str = "emotion",
                           domains: list[str] | None = None) -> list[dict]:
        """Fit the head for every unit, keep the fitted models + eval data.

        Feeds the faithfulness experiments (formula swap, weight vs.
        empirical correlation).
        """
        domains = domains or DOMAINS
        store = []
        for fold in self.split.folds():
            feats = self.backbone.features_for_fold(fold.index)
            for dom in domains:
                _, _, _, meds = self.shared_context(fold, dom, feats)
                med = meds[mediator]
                for unit in self.iter_units(fold, dom, feats):
                    M_tr = med.transform(unit.X_train)
                    M_ev = med.transform(unit.X_eval)
                    head = make_head("ridge", self.cfg).fit(M_tr, unit.y_train)
                    store.append(dict(fold=unit.fold, domain=unit.domain,
                                      user_id=unit.user_id, head=head,
                                      M_train=M_tr, M_eval=M_ev, y_eval=unit.y_eval,
                                      E_eval=unit.E_eval))
            print(f"  fold {fold.index} done ({len(store)} units)", flush=True)
        return store
