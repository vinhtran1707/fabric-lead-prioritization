"""Unit tests for the modeling helpers (pure pandas/scikit-learn, no Spark)."""
import numpy as np
import pandas as pd

from leadscore import modeling as md


def _synthetic_gold(n_per_month=120, months=("2009-01-01", "2009-02-01", "2009-03-01", "2009-04-01")):
    rng = np.random.default_rng(0)
    rows = []
    rid = 0
    for m in months:
        for _ in range(n_per_month):
            rid += 1
            prev_success = rng.random() < 0.2
            rows.append({
                "lead_id": f"L{rid:06d}", "source_row_number": rid, "call_year": 2009,
                "call_month_start": pd.Timestamp(m),
                "job": rng.choice(["admin.", "student", "retired"]), "marital": "married",
                "education": "university.degree", "credit_default": "no", "housing_loan": "yes",
                "personal_loan": "no", "contact_channel": rng.choice(["cellular", "telephone"]),
                "call_month": pd.Timestamp(m).strftime("%b").lower(), "call_day_of_week": "mon",
                "previous_outcome": "success" if prev_success else "nonexistent",
                "age": int(rng.integers(20, 70)), "contacts_this_campaign": int(rng.integers(1, 5)),
                "days_since_previous_contact": 5.0 if prev_success else np.nan,
                "previous_contacts": int(prev_success), "was_previously_contacted": int(prev_success),
                "emp_var_rate": 1.1, "cons_price_idx": 93.9, "cons_conf_idx": -36.4,
                "euribor_3m": 4.8, "nr_employed": 5191.0,
                "converted": int(rng.random() < (0.7 if prev_success else 0.1)),
            })
    return pd.DataFrame(rows)


def test_top_share_capture():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    score = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05])
    assert md.top_share_capture(y, score, share=0.2) == 1 / 3   # top 2 leads hold 1 of 3 conversions
    assert md.top_share_capture(y, score, share=0.3) == 2 / 3


def test_monthly_splits_never_train_on_the_future():
    months = pd.Series(pd.to_datetime(["2009-02-01", "2009-03-01", "2009-03-01", "2009-04-01"]))
    for month, train, test in md.monthly_splits(months, start=pd.Timestamp("2009-03-01")):
        assert months[train].max() < month
        assert (months[test] == month).all()


def test_scoring_ranks_and_explains_leads():
    gold = _synthetic_gold()
    scored = md.score_month(gold, pd.Timestamp("2009-04-01"))
    assert len(scored) == 120
    assert scored["rank_in_month"].tolist() == list(range(1, 121))
    assert scored["priority_decile"].between(1, 10).all()
    # the planted signal (past success) should appear among the top lead's reasons
    assert any("Previous campaign outcome: success" in (r or "") or "earlier campaign" in (r or "")
               for r in scored.iloc[0][["reason_1", "reason_2", "reason_3"]])
    # economic indicators are constant within the month, so they never appear as reasons
    reasons = scored[["reason_1", "reason_2", "reason_3"]].stack().dropna().astype(str).tolist()
    assert not any(r.startswith(("3-month Euribor", "Employment", "Consumer")) for r in reasons)


def test_backtest_beats_chance_on_planted_signal():
    bt = md.run_backtest(_synthetic_gold(months=("2009-01-01", "2009-02-01", "2009-03-01", "2009-04-01", "2009-05-01")))
    champ = bt.summary.set_index("model").loc["logistic_regression"]
    assert champ["auc"] > 0.7
    assert set(bt.monthly["model"]) == {"logistic_regression", "lightgbm", "seller_rule", "logistic_never_retrained"}
