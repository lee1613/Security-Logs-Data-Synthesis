"""Day-4 train/eval harness: fit a detector, score it, return AUC-PR.

This is a MEASURING INSTRUMENT for synthetic-data quality, not a detection-science
contribution. Absolute detection numbers are not the headline and are never tuned
for: there is NO hyperparameter search, no early stopping on eval data, no
threshold tuning. The GBT params are fixed in `day4.gbt` and stay fixed, because
a search would silently tune on the holdout and destroy the experiment. A weak
number here is a finding to report, not a number to improve.

AUC-PR (average precision), not ROC-AUC. Evaluation is 51 positives against up to
204,228 negatives (~1:4000); at that base rate ROC-AUC is flatteringly useless.
The null value of average precision is the POSITIVE RATE, so ~0.00025 is chance
here, not 0.5.

Extreme imbalance is the normal case, not an edge case, so both paths are class
weighted: `class_weight="balanced"` for logistic, and `compute_sample_weight` at
fit time for the GBT (sklearn's GradientBoostingClassifier has no class_weight).
Unweighted, either collapses to predicting all-negative.

Only the logistic path standardizes. `dst_in_degree`/`src_out_degree`/
`n_hosts_for_cred` are raw degree counts on a heavy tail (means ~10,203 vs ~129
across corpora) -- unscaled, logistic regression is garbage. Trees are
scale-invariant, so scaling the GBT would be pure noise. The scaler is fit inside
the Pipeline on train_X only and never sees eval_X. eval_X/eval_y are touched
exactly once, to score.

No I/O, no plotting, no data loading beyond reading its own default params:
figures belong to scripts/run_day4.py, features to src/features.py.
"""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.config import load_config

MODELS = ("gbt", "logistic")


def _estimator(model, seed, params):
    if model == "gbt":
        # ponytail: config re-read per call (~1ms against a multi-second fit).
        # Memoize only if the sweep ever profiles as config-bound.
        p = load_config()["day4"]["gbt"] if params is None else params
        return GradientBoostingClassifier(random_state=seed, **p)
    if model == "logistic":
        # used exactly once, for legibility (coefficient signs), not for
        # competitive performance
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000,
                               random_state=seed, **(params or {})))
    raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")


def train_and_eval(train_X, train_y, eval_X, eval_y, *, model="gbt", seed,
                   params=None):
    """Class-weighted fit on train, scored once on eval.
    Returns {"auc_pr": float, "pr_curve": (precision, recall), "model": fitted}.
    `params` overrides the config defaults (GBT only); same seed + same data
    gives a bitwise-identical auc_pr."""
    est = _estimator(model, seed, params)
    if model == "gbt":
        est.fit(train_X, train_y,
                sample_weight=compute_sample_weight("balanced", train_y))
    else:
        est.fit(train_X, train_y)   # class_weight="balanced" handles it

    scores = est.predict_proba(eval_X)[:, 1]
    precision, recall, _ = precision_recall_curve(eval_y, scores)
    return {"auc_pr": float(average_precision_score(eval_y, scores)),
            "pr_curve": (precision, recall),
            "model": est}
