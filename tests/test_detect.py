import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import src.detect as D

MODELS = ("gbt", "logistic")


def _separable(n=400, rng=None):
    """Two clean blobs. Any working classifier should get AUC-PR ~= 1.0."""
    rng = rng or np.random.default_rng(0)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    X = rng.normal(size=(n, 3)) + y[:, None] * 8.0
    return X, y


def _noise(n=2000, base_rate=0.1, rng=None):
    """Labels independent of features. The correct null for average precision
    is the positive rate, not 0.5 -- that is what this pins."""
    rng = rng or np.random.default_rng(1)
    X = rng.normal(size=(n, 3))
    y = (rng.random(n) < base_rate).astype(int)
    return X, y


def _imbalanced(n_neg=10000, n_pos=50, rng=None):
    """1:200 with a real but modest (overlapping) signal -- the shape of the
    actual problem (~650 positives vs ~484k negatives). Unweighted, a GBT here
    recovers almost none of the positives; measured over 30 seeds, unweighted
    recall never exceeded 0.08 while the weighted fit never fell below 0.42.
    Those measurements set the thresholds in the weighting test."""
    rng = rng or np.random.default_rng(2)
    X = np.vstack([rng.normal(size=(n_neg, 3)),
                   rng.normal(size=(n_pos, 3)) + 1.0])
    y = np.array([0] * n_neg + [1] * n_pos)
    return X, y


# fast params so the suite does not pay for 100 trees on toy fixtures; the real
# runs read day4.gbt from config (exercised by test_default_params_come_from_config).
# subsample MUST stay here: without it the GBT is deterministic regardless of
# random_state and the different-seed assertion silently proves nothing.
FAST = {"n_estimators": 25, "max_depth": 3, "subsample": 0.8}


def _run(train, evl, **kw):
    kw.setdefault("seed", 0)
    if kw.get("model", "gbt") == "gbt":
        kw.setdefault("params", FAST)
    return D.train_and_eval(train[0], train[1], evl[0], evl[1], **kw)


@pytest.mark.parametrize("model", MODELS)
def test_separable_fixture_is_near_perfect(model):
    tr, ev = _separable(rng=np.random.default_rng(10)), _separable(rng=np.random.default_rng(11))
    out = _run(tr, ev, model=model)
    assert out["auc_pr"] > 0.99


@pytest.mark.parametrize("model", MODELS)
def test_pure_noise_lands_near_the_base_rate(model):
    """AUC-PR of a useless ranker is the positive rate. Compared against the
    ACTUAL eval base rate, not the nominal 0.10. Tolerance +-0.05 is wide enough
    for finite-sample wobble at ~200 positives, narrow enough that 'perfect'
    (1.0) and the ROC-chance value (0.5) both fail."""
    tr, ev = _noise(rng=np.random.default_rng(20)), _noise(rng=np.random.default_rng(21))
    out = _run(tr, ev, model=model)
    assert out["auc_pr"] == pytest.approx(ev[1].mean(), abs=0.05)


def test_same_seed_is_bitwise_identical_and_different_seed_is_not():
    tr, ev = _noise(rng=np.random.default_rng(30)), _noise(rng=np.random.default_rng(31))
    a = _run(tr, ev, seed=7)["auc_pr"]
    b = _run(tr, ev, seed=7)["auc_pr"]
    c = _run(tr, ev, seed=99)["auc_pr"]
    assert a == b                       # exact, not approx
    assert a != c                       # or the seed is being ignored


def test_class_weighting_prevents_collapse_to_all_negative():
    """The unweighted control fit is the point: same data, same params, no
    sample_weight. It does not literally predict one class -- it flags a handful
    of random negatives -- but it recovers essentially NO true positives, which is
    the collapse that matters at a 1:200 base rate. If train_and_eval scored the
    same, weighting would not be reaching the estimator."""
    (tX, ty), (eX, ey) = _imbalanced(rng=np.random.default_rng(40)), _imbalanced(rng=np.random.default_rng(41))

    control = GradientBoostingClassifier(random_state=0, **FAST).fit(tX, ty)
    control_recall = control.predict(eX)[ey == 1].mean()
    assert control_recall < 0.15, "fixture no longer collapses unweighted"

    pred = _run((tX, ty), (eX, ey))["model"].predict(eX)
    assert set(np.unique(pred)) == {0, 1}
    assert pred[ey == 1].mean() > 0.35        # measured min 0.42 over 30 seeds
    assert pred[ey == 1].mean() > control_recall + 0.25
    # and it is not buying that recall by flagging everything
    assert pred.mean() < 0.5


def test_logistic_is_weighted_too():
    """Same control construction as the GBT case -- asserting only that both
    classes appear would pass unweighted too, so it must be a comparison.
    Measured over 15 seeds: unweighted recall <= 0.06, weighted >= 0.68."""
    (tX, ty), (eX, ey) = _imbalanced(rng=np.random.default_rng(50)), _imbalanced(rng=np.random.default_rng(51))

    control = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    control_recall = control.fit(tX, ty).predict(eX)[ey == 1].mean()
    assert control_recall < 0.15, "fixture no longer collapses unweighted"

    pred = _run((tX, ty), (eX, ey), model="logistic")["model"].predict(eX)
    assert set(np.unique(pred)) == {0, 1}
    assert pred[ey == 1].mean() > 0.5
    assert pred.mean() < 0.5


def test_logistic_standardizes_but_gbt_does_not():
    """A degree-count-like column on a heavy tail: unscaled logistic is garbage.
    Blowing one feature up by 1e5 must not move the logistic result."""
    (tX, ty), (eX, ey) = _separable(rng=np.random.default_rng(61)), _separable(rng=np.random.default_rng(62))
    tX, eX = tX.copy(), eX.copy()
    tX[:, 0] *= 1e5
    eX[:, 0] *= 1e5
    assert _run((tX, ty), (eX, ey), model="logistic")["auc_pr"] > 0.99

    fitted = _run((tX, ty), (eX, ey), model="logistic")["model"]
    assert any("scaler" in name for name, _ in fitted.steps)
    assert not hasattr(_run((tX, ty), (eX, ey))["model"], "steps")   # gbt is bare


def test_scaler_never_sees_eval_data():
    """The scaler must be fit on train only. Replacing eval with values from a
    wildly different distribution must not change the fitted scaler's stats."""
    tr = _separable(rng=np.random.default_rng(70))
    small = _separable(rng=np.random.default_rng(71))
    huge = (small[0] * 1000 + 5e4, small[1])

    a = _run(tr, small, model="logistic")["model"].named_steps["standardscaler"]
    b = _run(tr, huge, model="logistic")["model"].named_steps["standardscaler"]
    assert np.array_equal(a.mean_, b.mean_)
    assert np.array_equal(a.scale_, b.scale_)


def test_unknown_model_raises():
    tr = ev = _separable()
    with pytest.raises(ValueError, match="unknown model"):
        D.train_and_eval(tr[0], tr[1], ev[0], ev[1], model="randomforest", seed=0)


def test_contract_keys_exactly():
    tr = ev = _separable()
    out = _run(tr, ev)
    assert set(out) == {"auc_pr", "pr_curve", "model"}
    assert isinstance(out["auc_pr"], float)


@pytest.mark.parametrize("model", MODELS)
def test_pr_curve_shape_and_monotonicity(model):
    tr, ev = _separable(rng=np.random.default_rng(80)), _noise(rng=np.random.default_rng(81))
    precision, recall = _run(tr, ev, model=model)["pr_curve"]

    assert len(precision) == len(recall)
    assert ((precision >= 0) & (precision <= 1)).all()
    assert ((recall >= 0) & (recall <= 1)).all()
    assert (np.diff(recall) <= 0).all()      # sklearn returns recall descending
    assert recall[0] == 1.0 and recall[-1] == 0.0


def test_default_params_come_from_config():
    """No params passed -> day4.gbt from config.yaml, unmodified."""
    from src.config import load_config
    tr = ev = _separable(n=40)
    fitted = D.train_and_eval(tr[0], tr[1], ev[0], ev[1], seed=0)["model"]
    for k, v in load_config()["day4"]["gbt"].items():
        assert fitted.get_params()[k] == v
