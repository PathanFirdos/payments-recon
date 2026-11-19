# Payments reconciliation: October 2026 month-end

Reconciles a payments platform's transactions against bank settlement credits, finds every
gap, and explains why. Python, pandas, pytest, Streamlit. All money is `decimal.Decimal`
(strings in CSV, never floats; the engine rejects floats).

## Run

```bash
pip install -r requirements.txt
python generate_data.py          # writes data/ (seed 42)
python reconcile.py data         # text report
streamlit run app.py             # UI: summary, equation, exception table
pytest -v                        # tests (regenerate data in a temp dir)
```

## Files

| File | Purpose |
|---|---|
| `generate_data.py` | Seeded data: 250 platform rows, 18 settlement batches, 19 bank rows (18 credits + 1 duplicate), plus `ground_truth.csv` |
| `reconcile.py` | The reconciliation engine |
| `app.py` | Streamlit UI (Summary A, Exceptions B, equation, batch detail) |
| `tests/test_reconcile.py` | pytest suite; the only code that reads `ground_truth.csv` |

`reconcile.py` and `app.py` never read `ground_truth.csv`; a test greps both files to keep it that way.

## Assumptions

1. INR only. A non-INR row raises an error.
2. Reporting period is Oct 1-31, 2026. Platform rows dated outside it are ignored (and counted under "ignored").
3. Settlement is T+1 to T+2 business days, skipping weekends only (no bank holidays). `expected_settlement_date` is taken from the data and is the deadline.
4. One bank credit per settlement batch. If a batch somehow has several distinct credits they are summed and compared.
5. Gateway fee is 1.5%, rounded to 2 decimals with `ROUND_HALF_UP`, charged on successful payments only. Refunds do not reverse fees. The engine recomputes the fee from the amount; it does not trust the `gateway_fee_amount` column.
6. Expected batch net = successful payments - valid refunds - fee (sum of per-transaction rounded fees). FAILED rows are excluded.
7. Duplicate bank row = same `bank_transaction_id` AND same `utr_number`. The first row is kept; duplicates are removed before any batch comparison or total. Near-duplicates are out of scope and kept.
8. Tolerance: difference 0.00 = MATCHED. Exactly 0.01 is ROUNDING only with proof: the fee recomputed both ways (sum of per-transaction fees vs one fee on the batch total) must differ by that same signed 0.01. Otherwise UNEXPLAINED. Anything above 0.01 is UNEXPLAINED.
9. A refund is valid if its `original_transaction_id` exists as a PAYMENT in the platform data. Otherwise it is ORPHAN_REFUND: excluded from batch calculations and reported separately as a data-quality exception.
10. A blank `settlement_batch_id` is recovered by `merchant_id` + `expected_settlement_date` against batches already known from other rows. Exactly one match is assigned (FALLBACK_MATCH, not a gap). Zero or several matches give MANUAL_REVIEW: the row is not assigned and not counted in expected settlement. Any batch difference that causes stays UNEXPLAINED until someone assigns it.
11. Late settlement: expected settlement after the cut-off and the credit arrives later. It shows as PENDING / in transit, never missing. POTENTIALLY_MISSING only applies when the expected date is on or before the cut-off and no credit was seen.
12. Cut-off view (default): bank credits dated after Oct 31 are not part of the totals. Tick "Include November bank credits" (or `include_post_period_credits=True`) and the late batches match.

## Sign convention and the equation

`difference = expected - bank`. Positive means the bank paid less than expected.

```
Total difference = Net explained difference + Unexplained difference
```

Each gap contributes its net effect: late settlement = its credit net of fee (not gross), rounding = the 0.01,
orphan refund = 0.00, duplicate = its excess. Gross/raw amounts appear only in the exception table.

**Duplicate handling.** The spec asks for duplicates to be removed before any total, and also for the
duplicate to enter the equation as a negative excess. Both cannot hold in one number, so the app shows two
views, each balancing with unexplained = 0.00:

* **Main view** (bank total after removing duplicates): the duplicate's effect is 0.00 because it was removed.
* **Statement view** (bank credits exactly as received): the duplicate enters as its excess, negative, because the bank total was too high.

## Data notes

Seed 42. Planted gaps: GT001/GT002 late settlements (B017/B018), GT003 rounding (B007), GT004 duplicate bank
entry (BNK5012, B012), GT005 orphan refund (RFND3001 -> TXN9999). Non-exceptions: FALLBACK_MATCH for
TXN1201-TXN1204 and EXCLUDED_FAILED for TXN1301-TXN1303. Normal-batch amounts are multiples of INR 2.00 so
per-transaction and batch-level fees agree exactly; only B007 differs. The generator builds bank credits
independently of `reconcile.py` (one fee on the batch total, as a bank would).
