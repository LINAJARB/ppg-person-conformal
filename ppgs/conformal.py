"""Conformal prediction intervals calibrated per person.

Three interval families, all built on the same three descriptors:
  constant    score |e|                      interval  y_hat +/- Q
  normalized  score |e| / sigma(x)           interval  y_hat +/- Q sigma(x)
              with log sigma(x) a linear function of the descriptors,
              fitted by least squares on log(|e| + 1)
  cqr         linear quantile regression of the signed error e = y - y_hat
              (levels alpha/2 and 1 - alpha/2), then conformal correction

Two ways to weight the calibration scores:
  window      every window counts the same (standard split conformal)
  subject     every person counts the same: the windows of a person share
              a total weight of 1 (hierarchical conformal prediction)

Every interval is intersected with the range of the estimator grid, 40-200 bpm.
This only affects references outside that range (4 GalaxyPPG windows below 40 bpm).

Evaluation is leave-one-subject-out. For the held-out person, the scores of
the other persons are obtained out of fold (5 folds of persons), so that no
score comes from a model that has seen that person (cross-fitting).
"""

import numpy as np
from sklearn.linear_model import LinearRegression, QuantileRegressor

from .estimator import FEATURES

ALPHA = 0.10
N_FOLDS = 5
CQR_MAX_FIT = 8000  # windows used to fit each quantile regression
HR_RANGE = (40.0, 200.0)  # range of the estimator grid: intervals are intersected with it


class Normalized:
    def fit(self, X, e):
        self.lr = LinearRegression().fit(X, np.log(np.abs(e) + 1.0))
        return self

    def sigma(self, X):
        return np.exp(self.lr.predict(X))

    def score(self, X, e):
        return np.abs(e) / self.sigma(X)

    def interval(self, X, q):
        s = self.sigma(X)
        return -q * s, q * s


class Constant:
    def fit(self, X, e):
        return self

    def score(self, X, e):
        return np.abs(e)

    def interval(self, X, q):
        return -q * np.ones(len(X)), q * np.ones(len(X))


class CQR:
    def fit(self, X, e, rng=np.random.default_rng(0)):
        idx = rng.choice(len(X), min(len(X), CQR_MAX_FIT), replace=False)
        self.lo = QuantileRegressor(quantile=ALPHA / 2, alpha=0.0, solver="highs").fit(X[idx], e[idx])
        self.hi = QuantileRegressor(quantile=1 - ALPHA / 2, alpha=0.0, solver="highs").fit(X[idx], e[idx])
        return self

    def score(self, X, e):
        return np.maximum(self.lo.predict(X) - e, e - self.hi.predict(X))

    def interval(self, X, q):
        return self.lo.predict(X) - q, self.hi.predict(X) + q


METHODS = {"constant": Constant, "normalized": Normalized, "cqr": CQR}


def weighted_quantile(scores, weights, level):
    order = np.argsort(scores)
    cw = np.cumsum(weights[order]) / weights.sum()
    return scores[order][min(np.searchsorted(cw, level), len(scores) - 1)]


def calibration_quantile(scores, subjects, weighting):
    if weighting == "window":
        n = len(scores)
        return weighted_quantile(scores, np.ones(n), min(1.0, (1 - ALPHA) * (1 + 1 / n)))
    _, inv, counts = np.unique(subjects, return_inverse=True, return_counts=True)
    m = len(counts)
    return weighted_quantile(scores, 1.0 / counts[inv], min(1.0, (1 - ALPHA) * (1 + 1 / m)))


def person_folds(subjects, datasets, n_folds=N_FOLDS):
    """Assign persons to folds, alternating within each dataset (stratified)."""
    fold, k = {}, 0
    for ds in sorted(set(datasets)):
        for s in sorted(set(subjects[datasets == ds]), key=lambda v: int(v[1:])):
            fold[s] = k % n_folds
            k += 1
    return np.array([fold[s] for s in subjects])


def loso(df, method, weightings=("window", "subject")):
    """Leave-one-subject-out intervals. Returns a copy of df with columns
    lo_<weighting>, hi_<weighting> (absolute HR bounds)."""
    X = df[FEATURES].to_numpy()
    e = (df.hr_ref - df.hr_hat).to_numpy()
    subj, ds = df.subject.to_numpy(), df.dataset.to_numpy()
    out = df.copy()
    for w in weightings:
        out[f"lo_{w}"] = np.nan
        out[f"hi_{w}"] = np.nan
    for t in np.unique(subj):
        test, train = subj == t, subj != t
        folds = person_folds(subj[train], ds[train])
        oof = np.empty(train.sum())
        Xtr, etr = X[train], e[train]
        for k in range(N_FOLDS):
            f = folds == k
            oof[f] = METHODS[method]().fit(Xtr[~f], etr[~f]).score(Xtr[f], etr[f])
        model = METHODS[method]().fit(Xtr, etr)
        for w in weightings:
            q = calibration_quantile(oof, subj[train], w)
            lo, hi = model.interval(X[test], q)
            out.loc[test, f"lo_{w}"] = np.maximum(df.hr_hat[test] + lo, HR_RANGE[0])
            out.loc[test, f"hi_{w}"] = np.minimum(df.hr_hat[test] + hi, HR_RANGE[1])
    return out


def summarize(res, w):
    covered = (res.hr_ref >= res[f"lo_{w}"]) & (res.hr_ref <= res[f"hi_{w}"])
    width = res[f"hi_{w}"] - res[f"lo_{w}"]
    per_subject = covered.groupby(res.subject).mean()
    return {
        "coverage (windows)": covered.mean(),
        "coverage (mean over persons)": per_subject.mean(),
        "persons at target": f"{(per_subject >= 1 - ALPHA).sum()}/{len(per_subject)}",
        "worst person": per_subject.min(),
        "mean width (bpm)": width.mean(),
        "median width (bpm)": width.median(),
    }


def calibrate_and_apply(dev, test, method, weightings=("window", "subject")):
    """Calibrate on all development persons, apply once to a new dataset.

    Scores of the development persons are obtained out of fold (5 folds of
    persons), the final model is fitted on all of them, and the intervals
    are computed for every window of the test set."""
    X, e = dev[FEATURES].to_numpy(), (dev.hr_ref - dev.hr_hat).to_numpy()
    subj, ds = dev.subject.to_numpy(), dev.dataset.to_numpy()
    folds = person_folds(subj, ds)
    oof = np.empty(len(dev))
    for k in range(N_FOLDS):
        f = folds == k
        oof[f] = METHODS[method]().fit(X[~f], e[~f]).score(X[f], e[f])
    model = METHODS[method]().fit(X, e)
    out = test.copy()
    for w in weightings:
        lo, hi = model.interval(test[FEATURES].to_numpy(), calibration_quantile(oof, subj, w))
        out[f"lo_{w}"] = np.maximum(test.hr_hat + lo, HR_RANGE[0])
        out[f"hi_{w}"] = np.minimum(test.hr_hat + hi, HR_RANGE[1])
    return out
