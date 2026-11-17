"""Lead conversion modeling: time-based backtest, champion selection, scoring, reason codes.

Design choices (each backed by an experiment recorded in docs/model_card.md):
- Validation is a rolling monthly backtest: for every month, train on all earlier
  months and score that month. Random splits would leak future market conditions.
- The model is retrained every month on an expanding window. A model trained once
  on 2008 data degrades toward a coin flip as the market shifts.
- Logistic regression is the champion. LightGBM matched it but did not beat it, so
  the simpler, directly explainable model wins. Both are compared against the
  seller's rule of thumb ("call past successes first").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
FIRST_BACKTEST_MONTH = pd.Timestamp("2009-03-01")
TOP_SHARE = 0.20  # sellers' daily capacity: call the top 20% of leads

FEATURE_LABELS = {
    "job": "Job", "marital": "Marital status", "education": "Education",
    "credit_default": "Credit default", "housing_loan": "Housing loan",
    "personal_loan": "Personal loan", "contact_channel": "Contact channel",
    "call_month": "Call month", "call_day_of_week": "Call day",
    "previous_outcome": "Previous campaign outcome", "age": "Age",
    "contacts_this_campaign": "Contacts this campaign",
    "days_since_previous_contact": "Days since last contact",
    "previous_contacts": "Previous contacts", "was_previously_contacted": "Contacted before",
    "emp_var_rate": "Employment variation rate", "cons_price_idx": "Consumer price index",
    "cons_conf_idx": "Consumer confidence", "euribor_3m": "3-month Euribor rate",
    "nr_employed": "Employment level",
}


# --------------------------------------------------------------------------- models
def make_logistic() -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC_FEATURES),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, C=0.5))])


def make_lightgbm():
    import lightgbm as lgb  # imported lazily so the champion path has no extra dependency

    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                              min_child_samples=50, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=7, verbose=-1)


def _as_lgbm_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[FEATURES].copy()
    for c in CATEGORICAL_FEATURES:
        out[c] = out[c].astype("category")
    return out


def seller_rule_score(df: pd.DataFrame) -> np.ndarray:
    """Rule of thumb: past successes first, then anyone contacted before."""
    return ((df["previous_outcome"] == "success").astype(int) * 2
            + df["was_previously_contacted"].fillna(0).astype(int)).to_numpy(dtype=float)


# --------------------------------------------------------------------------- metrics
def top_share_capture(y: pd.Series | np.ndarray, score: np.ndarray, share: float = TOP_SHARE) -> float:
    """Share of all conversions found by calling only the top `share` of leads.

    Ties are broken by original order so the rule baseline is not flattered.
    """
    y = np.asarray(y)
    if y.sum() == 0:
        return float("nan")
    k = max(1, int(round(len(score) * share)))
    top = np.argsort(-np.asarray(score), kind="stable")[:k]
    return float(y[top].sum() / y.sum())


def monthly_splits(months: pd.Series, start: pd.Timestamp = FIRST_BACKTEST_MONTH):
    """Yield (month, train_mask, test_mask) with training strictly before the test month."""
    for month in sorted(m for m in months.unique() if m >= start):
        yield month, (months < month).to_numpy(), (months == month).to_numpy()


# --------------------------------------------------------------------------- backtest
@dataclass
class BacktestResult:
    monthly: pd.DataFrame       # one row per month x model
    predictions: pd.DataFrame   # champion out-of-time predictions per lead
    summary: pd.DataFrame       # one row per model, weighted by month size


def run_backtest(gold: pd.DataFrame) -> BacktestResult:
    df = gold.sort_values("source_row_number").reset_index(drop=True)
    months = pd.to_datetime(df["call_month_start"])
    y = df["converted"]
    rows, preds = [], []
    # Drift baseline: one model trained on everything before the backtest and never retrained.
    frozen_train = (months < FIRST_BACKTEST_MONTH).to_numpy()
    frozen = make_logistic().fit(df.loc[frozen_train, FEATURES], y[frozen_train])
    for month, train, test in monthly_splits(months):
        if y[test].nunique() < 2 or y[train].nunique() < 2:
            continue
        champion = make_logistic().fit(df.loc[train, FEATURES], y[train])
        challenger = make_lightgbm().fit(_as_lgbm_frame(df.loc[train]), y[train])
        scores = {
            "logistic_regression": champion.predict_proba(df.loc[test, FEATURES])[:, 1],
            "lightgbm": challenger.predict_proba(_as_lgbm_frame(df.loc[test]))[:, 1],
            "logistic_never_retrained": frozen.predict_proba(df.loc[test, FEATURES])[:, 1],
            "seller_rule": seller_rule_score(df.loc[test]),
        }
        yt = y[test].to_numpy()
        for name, s in scores.items():
            is_prob = name != "seller_rule"
            rows.append({
                "month": month, "model": name, "leads": int(test.sum()),
                "conversion_rate": float(yt.mean()),
                "auc": float(roc_auc_score(yt, s)),
                "avg_precision": float(average_precision_score(yt, s)),
                "top20_capture": top_share_capture(yt, s),
                "brier": float(np.mean((s - yt) ** 2)) if is_prob else np.nan,
                "predicted_rate": float(s.mean()) if is_prob else np.nan,
            })
        preds.append(pd.DataFrame({
            "lead_id": df.loc[test, "lead_id"].to_numpy(), "month": month,
            "score": scores["logistic_regression"], "converted": yt,
        }))
    monthly = pd.DataFrame(rows)
    w = lambda g, c: np.average(g[c], weights=g["leads"])  # noqa: E731
    summary = (monthly.groupby("model")
               .apply(lambda g: pd.Series({
                   "months": len(g), "auc": w(g, "auc"), "avg_precision": w(g, "avg_precision"),
                   "top20_capture": w(g, "top20_capture"),
                   "brier": w(g.dropna(subset=["brier"]), "brier") if g["brier"].notna().any() else np.nan,
               }), include_groups=False)
               .reset_index().sort_values("auc", ascending=False))
    return BacktestResult(monthly, pd.concat(preds, ignore_index=True), summary)


def calibration_table(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Predicted vs actual conversion by score decile (pooled out-of-time predictions)."""
    p = predictions.copy()
    p["decile"] = pd.qcut(p["score"].rank(method="first"), bins, labels=False) + 1
    return (p.groupby("decile")
            .agg(leads=("score", "size"), avg_predicted=("score", "mean"), actual_rate=("converted", "mean"))
            .reset_index())


# --------------------------------------------------------------------------- scoring + reasons
def reason_codes(model: Pipeline, rows: pd.DataFrame, reference: pd.DataFrame, top_n: int = 3) -> list[list[str]]:
    """Top drivers pushing each lead's score above its peers, from the logistic model.

    Each transformed column's contribution is coef * (x - mean of `reference`), the
    exact SHAP value for a linear model. Passing the same month's leads as the
    reference answers the seller's question, "why is this lead above the others I
    could call this month?". Economic indicators are identical for everyone in a
    month, so they cancel out instead of crowding every lead's reasons. One-hot
    columns are summed back to their source feature ("Previous campaign outcome:
    success").
    """
    pre, clf = model.named_steps["pre"], model.named_steps["clf"]
    x = pre.transform(rows[FEATURES])
    base = np.asarray(pre.transform(reference[FEATURES]).mean(axis=0)).ravel()
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    contrib = (x - base) * clf.coef_.ravel()

    names = pre.get_feature_names_out()
    source = [n.split("__", 1)[1] for n in names]
    source = [next(f for f in FEATURES if s == f or s.startswith(f + "_")) for s in source]
    by_feature = pd.DataFrame(contrib, columns=source).T.groupby(level=0).sum().T

    out = []
    for i, (_, row) in enumerate(rows.reset_index(drop=True).iterrows()):
        top = by_feature.iloc[i].sort_values(ascending=False)
        top = top[top > 1e-9].head(top_n)
        out.append([_reason_text(f, row[f]) for f in top.index])
    return out


def _reason_text(feature: str, value) -> str:
    if feature == "was_previously_contacted":
        return "Contacted in an earlier campaign" if value == 1 else "New lead, not contacted before"
    return f"{FEATURE_LABELS.get(feature, feature)}: {_fmt(value)}"


def _fmt(v) -> str:
    if isinstance(v, float):
        return "unknown" if np.isnan(v) else f"{v:g}"
    return str(v)


def score_month(gold: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    """Score one month's leads with a champion trained on all earlier months."""
    df = gold.copy()
    months = pd.to_datetime(df["call_month_start"])
    train, target = df[months < month], df[months == month]
    model = make_logistic().fit(train[FEATURES], train["converted"])
    score = model.predict_proba(target[FEATURES])[:, 1]
    reasons = reason_codes(model, target, reference=target)
    out = pd.DataFrame({
        "lead_id": target["lead_id"].to_numpy(), "score_month": month, "score": score,
        "converted": target["converted"].to_numpy(),
    })
    out["rank_in_month"] = out["score"].rank(ascending=False, method="first").astype(int)
    out["priority_decile"] = (np.ceil(out["rank_in_month"] / len(out) * 10)).astype(int)
    for k in range(3):
        out[f"reason_{k + 1}"] = [r[k] if len(r) > k else None for r in reasons]
    return out.sort_values("rank_in_month")


def lift_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Cumulative share of conversions captured by priority decile."""
    t = (scored.groupby("priority_decile")
         .agg(leads=("lead_id", "size"), conversions=("converted", "sum")).reset_index())
    t["cum_leads_share"] = t["leads"].cumsum() / t["leads"].sum()
    t["cum_conversions_share"] = t["conversions"].cumsum() / t["conversions"].sum()
    t["lift"] = t["cum_conversions_share"] / t["cum_leads_share"]
    return t
