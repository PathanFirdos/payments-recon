"""Streamlit UI for the October 2026 payments reconciliation.

Run:  streamlit run app.py
Reads data/platform_transactions.csv and data/bank_transactions.csv (or uploaded CSVs).
It never reads the test answer-key file.
"""
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from reconcile import PERIOD_END, load_bank, load_platform, reconcile

try:
    _BASE = Path(__file__).resolve().parent
except NameError:  # pragma: no cover
    _BASE = Path.cwd()
DATA_DIR = Path(os.environ.get("RECON_DATA_DIR", _BASE / "data"))


def money(x) -> str:
    return f"{x:,.2f}"


st.set_page_config(page_title="Payments reconciliation", layout="wide")
st.title("Payments reconciliation: October 2026 month-end")
st.caption("INR. difference = expected (platform) minus bank. Positive: bank paid less than expected.")

with st.sidebar:
    st.header("Options")
    include_nov = st.checkbox("Include November bank credits", value=False,
                              help="Off = cut-off view at the date below. On = late settlements match.")
    cutoff = st.date_input("Cut-off date", value=PERIOD_END)
    st.header("Data")
    up_platform = st.file_uploader("platform_transactions.csv", type="csv")
    up_bank = st.file_uploader("bank_transactions.csv", type="csv")

try:
    platform = (pd.read_csv(up_platform, dtype=str, keep_default_na=False)
                if up_platform else load_platform(DATA_DIR / "platform_transactions.csv"))
    bank = (pd.read_csv(up_bank, dtype=str, keep_default_na=False)
            if up_bank else load_bank(DATA_DIR / "bank_transactions.csv"))
except FileNotFoundError:
    st.error(f"No data found in {DATA_DIR}. Run `python generate_data.py` first, or upload both CSVs.")
    st.stop()

result = reconcile(platform, bank, period_end=cutoff, include_post_period_credits=include_nov)
s, eq = result.summary, result.equation

# ------------------------------------------------------------------ A. Summary
st.header("A. Summary")
c1, c2, c3 = st.columns(3)
c1.metric("Total platform payments", money(s["total_platform_payments"]))
c2.metric("Total expected settlement", money(s["total_expected_settlement"]),
          help="Successful payments minus valid refunds minus gateway fee. Failed payments and orphan refunds excluded.")
c3.metric("Total bank credits (after duplicates removed)", money(s["total_bank_credits"]))
c4, c5, c6 = st.columns(3)
c4.metric("Total difference (expected - bank)", money(s["total_difference"]))
c5.metric("Sum of explained gaps", money(s["net_explained"]))
c6.metric("Unexplained difference", money(s["unexplained"]))

if s["duplicates_removed"]:
    st.caption(f"Bank credits as received: {money(s['total_bank_credits_as_received'])}. "
               f"Duplicate rows removed before any comparison: {money(s['duplicates_removed'])}.")

st.subheader("Equation")
st.code(result.equation_text(), language="text")
if eq["holds"] and eq["statement_holds"] and eq["unexplained"] == 0:
    st.success("Total difference = net explained + unexplained, and unexplained is 0.00.")
elif eq["holds"] and eq["statement_holds"]:
    st.warning(f"The equation holds, but {money(eq['unexplained'])} is unexplained.")
else:
    st.error("The equation does not balance. Treat this run as unreliable.")

pending = [e for e in result.exceptions if e.gap_type == "LATE_SETTLEMENT"]
if pending:
    st.info("Pending / in transit (not missing): "
            + "; ".join(f"{e.batch_id} net {money(e.effect_dedup)}" for e in pending))

# ------------------------------------------------------------------ B. Exceptions
st.header("B. Exceptions")
if result.exceptions:
    st.dataframe(result.exceptions_df(), hide_index=True)
else:
    st.write("No exceptions.")

# ------------------------------------------------------------------ details
with st.expander("Batch-level results"):
    st.dataframe(result.batches_df(), hide_index=True)

with st.expander("Data-quality notes and non-exception outcomes"):
    st.markdown("**Fallback batch matches** (blank batch id recovered by merchant + expected date; not a gap)")
    st.dataframe(pd.DataFrame(
        [{"transaction_id": k, "assigned_batch": v} for k, v in result.fallback_assignments.items()],
        columns=["transaction_id", "assigned_batch"]), hide_index=True)
    st.markdown("**Failed rows excluded from expected settlement**")
    st.dataframe(pd.DataFrame(
        [{**f, "amount": money(f["amount"])} for f in result.excluded_failed],
        columns=["transaction_id", "type", "batch_id", "amount"]), hide_index=True)
    if result.manual_review:
        st.warning("Manual review needed: " + ", ".join(result.manual_review))
    st.caption(f"Ignored: {result.ignored}")
