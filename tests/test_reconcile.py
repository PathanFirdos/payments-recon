"""Tests for the payments reconciliation.

Only this file reads ground_truth.csv. Data is regenerated (seed 42) into a temp dir, so the
suite is hermetic.
"""
import csv
import re
import sys
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_data  # noqa: E402
from reconcile import (BANK_COLUMNS, PLATFORM_COLUMNS, load_bank, load_platform,  # noqa: E402
                       reconcile)

D = Decimal
BATCH_RE = re.compile(r"^B\d{3}$")


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("recon_data")
    generate_data.generate(d, seed=42)
    return d


@pytest.fixture(scope="session")
def platform(data_dir):
    return load_platform(data_dir / "platform_transactions.csv")


@pytest.fixture(scope="session")
def bank(data_dir):
    return load_bank(data_dir / "bank_transactions.csv")


@pytest.fixture(scope="session")
def gt(data_dir):
    with open(data_dir / "ground_truth.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["ids"] = r["affected_ids"].split(";")
        r["amount"] = D(r["expected_amount"])
    return {r["id"]: r for r in rows}


@pytest.fixture(scope="session")
def oct31(platform, bank):
    """Cut-off view: November bank credits not yet included."""
    return reconcile(platform, bank)


@pytest.fixture(scope="session")
def with_nov(platform, bank):
    return reconcile(platform, bank, include_post_period_credits=True)


def batch_of(row):
    return next(t for t in row["ids"] if BATCH_RE.match(t))


def rows_of(gt, gap_type):
    return [r for r in gt.values() if r["gap_type"] == gap_type]


def fee(amount):
    return (amount * D("0.015")).quantize(D("0.01"), rounding=ROUND_HALF_UP)


# ----------------------------------------------------------------- synthetic-data helpers
def P(tid, amount, batch="B001", merchant="M001", expected="2026-10-07", tdate="2026-10-05",
      status="SUCCESS", ttype="PAYMENT", original=""):
    return dict(transaction_id=tid, customer_id="C1", merchant_id=merchant, transaction_date=tdate,
                transaction_type=ttype, amount=amount, currency="INR", status=status,
                original_transaction_id=original, settlement_batch_id=batch,
                expected_settlement_date=expected, gateway_fee_amount="0.00")


def K(bank_id, batch, credit, date="2026-10-07", utr=None, merchant="M001"):
    return dict(bank_transaction_id=bank_id, utr_number=utr or "UTR" + bank_id, settlement_batch_id=batch,
                merchant_id=merchant, bank_date=date, credit_amount=credit, currency="INR",
                transaction_type="SETTLEMENT_CREDIT")


def frames(plat_rows, bank_rows):
    return (pd.DataFrame(plat_rows, columns=PLATFORM_COLUMNS),
            pd.DataFrame(bank_rows, columns=BANK_COLUMNS))


# ============================================================ spec tests 1-9 (ground truth)
@pytest.mark.parametrize("gt_id", ["GT001", "GT002"])
def test_01_02_late_settlement_pending_then_matched(gt_id, gt, oct31, with_nov):
    row = gt[gt_id]
    bid = batch_of(row)
    txn = next(t for t in row["ids"] if t.startswith("TXN"))
    bnk = next(t for t in row["ids"] if t.startswith("BNK"))

    # at the Oct 31 cut-off: pending / in transit, flagged as a late settlement, never "missing"
    b = oct31.batch(bid)
    assert b.status == "PENDING"
    late = [e for e in oct31.exceptions_of("LATE_SETTLEMENT") if e.batch_id == bid]
    assert len(late) == 1
    assert set(late[0].affected_ids) == set(row["ids"])
    assert late[0].amount == row["amount"]                  # gross shown in the exception table
    assert late[0].effect_dedup == row["amount"] - fee(row["amount"])   # net of fee in the equation
    assert txn in b.payment_ids and bnk in late[0].affected_ids
    assert not oct31.exceptions_of("POTENTIALLY_MISSING")

    # once the November credit is included the batch matches and the exception disappears
    m = with_nov.batch(bid)
    assert m.status == "MATCHED"
    assert m.difference == D("0.00")
    assert m.bank_ids == [bnk]
    assert not [e for e in with_nov.exceptions if e.batch_id == bid]


def test_03_rounding_batch_b007(gt, oct31):
    row = gt["GT003"]
    ex = oct31.exceptions_of("ROUNDING")
    assert len(ex) == 1
    assert set(ex[0].affected_ids) == set(row["ids"])
    assert ex[0].amount == row["amount"] == D("0.01")
    b = oct31.batch("B007")
    assert b.status == "ROUNDING" and b.difference == D("0.01")
    # the proof: the two fee methods differ by exactly that amount
    assert b.fee_on_total - b.fee_per_txn == D("0.01")
    assert "45.45" in ex[0].reason and "45.46" in ex[0].reason


def test_04_duplicate_bank_entry_counted_once(gt, bank, oct31):
    row = gt["GT004"]
    ex = oct31.exceptions_of("DUPLICATE_BANK_ENTRY")
    assert len(ex) == 1
    assert set(ex[0].affected_ids) == set(row["ids"])
    assert ex[0].amount == row["amount"] == D("4250.00")
    assert oct31.duplicates_removed == ["BNK5012"]
    assert (bank["bank_transaction_id"] == "BNK5012").sum() == 2        # really present twice in the file
    b = oct31.batch("B012")
    assert b.bank_ids == ["BNK5012"] and b.bank_credit == D("4250.00")  # counted once
    assert b.status == "MATCHED"
    assert oct31.summary["duplicates_removed"] == D("4250.00")
    assert (oct31.summary["total_bank_credits_as_received"] - oct31.summary["total_bank_credits"]
            == D("4250.00"))


def test_05_orphan_refund(gt, oct31):
    row = gt["GT005"]
    ex = oct31.exceptions_of("ORPHAN_REFUND")
    assert len(ex) == 1
    assert set(ex[0].affected_ids) == set(row["ids"]) == {"RFND3001", "TXN9999"}
    assert ex[0].amount == row["amount"] == D("300.00")
    assert ex[0].effect_dedup == D("0.00") and ex[0].effect_statement == D("0.00")
    assert oct31.orphan_refunds == ["RFND3001"]
    # excluded from the batch calculation: its batch still matches the bank to the cent
    assert "RFND3001" not in oct31.batch("B010").refund_ids
    assert oct31.batch("B010").status == "MATCHED"


def test_06_no_false_positives(gt, oct31):
    gap_rows = [r for r in gt.values() if r["gap_type"] in
                {"LATE_SETTLEMENT", "ROUNDING", "DUPLICATE_BANK_ENTRY", "ORPHAN_REFUND"}]
    planted_batches = {batch_of(r) for r in gap_rows if any(BATCH_RE.match(t) for t in r["ids"])}
    assert planted_batches == {"B007", "B012", "B017", "B018"}
    unaffected = [b for b in oct31.batches if b.batch_id not in planted_batches]
    assert len(unaffected) == 14
    assert all(b.status == "MATCHED" and b.difference == D("0.00") for b in unaffected)
    # the exception list is exactly the planted gaps: same types, same affected IDs
    got = sorted((e.gap_type, frozenset(e.affected_ids)) for e in oct31.exceptions)
    want = sorted((r["gap_type"], frozenset(r["ids"])) for r in gap_rows)
    assert got == want
    assert not [e for e in oct31.exceptions if e.gap_type in
                {"UNEXPLAINED", "POTENTIALLY_MISSING", "MANUAL_REVIEW", "UNMATCHED_BANK_CREDIT"}]


def test_07_failed_payments_excluded(gt, platform, oct31):
    rows = rows_of(gt, "EXCLUDED_FAILED")
    assert [r["ids"][0] for r in rows] == ["TXN1301", "TXN1302", "TXN1303"]
    in_batches = {t for b in oct31.batches for t in b.payment_ids}
    for r in rows:
        txn, bid = r["ids"]
        assert txn not in in_batches
        assert oct31.batch(bid).status == "MATCHED"     # bank credit was paid without the failed payment
    assert {f["transaction_id"] for f in oct31.excluded_failed} == {r["ids"][0] for r in rows}
    failed_total = sum((r["amount"] for r in rows), D(0))
    all_payments = sum((D(a) for a, t in zip(platform["amount"], platform["transaction_type"])
                        if t == "PAYMENT"), D(0))
    assert oct31.summary["total_platform_payments"] + failed_total == all_payments


def test_08_fallback_matching(gt, platform, oct31):
    rows = rows_of(gt, "FALLBACK_MATCH")
    assert [r["ids"][0] for r in rows] == ["TXN1201", "TXN1202", "TXN1203", "TXN1204"]
    flagged = {i for e in oct31.exceptions for i in e.affected_ids}
    for r in rows:
        txn, bid = r["ids"]
        assert platform.loc[platform["transaction_id"] == txn, "settlement_batch_id"].item() == ""
        assert oct31.fallback_assignments[txn] == bid
        assert txn in oct31.batch(bid).payment_ids
        assert oct31.batch(bid).status == "MATCHED"     # only possible if the batch was right
        assert txn not in flagged                       # not reported as a gap
    assert not oct31.manual_review


def test_09_equation(gt, oct31, with_nov):
    late_nets = sum((r["amount"] - fee(r["amount"]) for r in rows_of(gt, "LATE_SETTLEMENT")), D(0))
    rounding = gt["GT003"]["amount"]
    dup = gt["GT004"]["amount"]

    eq = oct31.equation
    s = oct31.summary
    assert eq["holds"]
    assert s["net_explained"] + s["unexplained"] == s["total_difference"]
    assert s["unexplained"] == D("0.00")
    assert s["total_difference"] == s["total_expected_settlement"] - s["total_bank_credits"]
    assert s["net_explained"] == late_nets + rounding == D("3644.51")
    assert eq["components"] == {"LATE_SETTLEMENT": D("3644.50"), "ROUNDING": D("0.01"),
                                "DUPLICATE_BANK_ENTRY": D("0.00"), "ORPHAN_REFUND": D("0.00")}

    # statement view: bank total as received, the duplicate counted and signed negative
    assert eq["statement_holds"] and eq["statement_unexplained"] == D("0.00")
    assert eq["statement_components"]["DUPLICATE_BANK_ENTRY"] == -dup
    assert eq["statement_total_difference"] == late_nets + rounding - dup == D("-605.49")
    assert eq["statement_total_difference"] == (s["total_expected_settlement"]
                                                - s["total_bank_credits_as_received"])

    # with November credits included only the rounding cent remains
    e2 = with_nov.equation
    assert e2["holds"] and e2["unexplained"] == D("0.00")
    assert with_nov.summary["total_difference"] == D("0.01") == with_nov.summary["net_explained"]
    assert e2["statement_total_difference"] == D("0.01") - dup


# ============================================================================ extra tests
def test_rounding_needs_proof_001_without_proof_is_unexplained():
    # one 100.00 payment: fee 1.50 either way, so a 0.01 gap has no rounding explanation
    p, b = frames([P("T1", "100.00")], [K("BNK1", "B001", "98.49")])
    r = reconcile(p, b)
    assert r.batch("B001").difference == D("0.01")
    assert r.batch("B001").status == "UNEXPLAINED"
    ex = r.exceptions_of("UNEXPLAINED")
    assert len(ex) == 1 and "without fee-rounding proof" in ex[0].reason
    assert r.equation["unexplained"] == D("0.01") and r.equation["holds"]


def test_rounding_with_proof_and_its_limits():
    pays = [P(f"T{i}", "1010.30") for i in range(3)]     # per-txn fees 45.45, batch fee 45.46
    ok = reconcile(*frames(pays, [K("BNK1", "B001", "2985.44")]))
    assert ok.batch("B001").status == "ROUNDING"
    too_big = reconcile(*frames(pays, [K("BNK1", "B001", "2985.43")]))        # 0.02: above tolerance
    assert too_big.batch("B001").status == "UNEXPLAINED"
    wrong_way = reconcile(*frames(pays, [K("BNK1", "B001", "2985.46")]))      # 0.01 the other way
    assert wrong_way.batch("B001").status == "UNEXPLAINED"
    exact = reconcile(*frames(pays, [K("BNK1", "B001", "2985.45")]))
    assert exact.batch("B001").status == "MATCHED"


def test_fee_rounds_half_up():
    from reconcile import gateway_fee
    assert gateway_fee(D("0.30")) == D("0.00")      # 0.0045 -> 0.00
    assert gateway_fee(D("0.50")) == D("0.01")      # 0.0075 -> 0.01
    assert gateway_fee(D("1010.30")) == D("15.15")  # 15.1545
    assert gateway_fee(D("3030.90")) == D("45.46")  # 45.4635


def test_several_fallback_candidates_give_manual_review():
    rows = [P("T1", "100.00", batch="B001"), P("T2", "200.00", batch="B002"),   # same merchant + date
            P("T3", "300.00", batch="")]
    r = reconcile(*frames(rows, [K("BNK1", "B001", "98.50"), K("BNK2", "B002", "197.00")]))
    assert r.manual_review == ["T3"]
    assert r.fallback_assignments == {}                  # never guess
    ex = r.exceptions_of("MANUAL_REVIEW")
    assert len(ex) == 1 and set(ex[0].affected_ids) == {"T3", "B001", "B002"}
    assert "T3" not in r.batch("B001").payment_ids + r.batch("B002").payment_ids


def test_single_fallback_candidate_is_assigned_and_zero_candidates_go_to_review():
    rows = [P("T1", "100.00", batch="B001"), P("T2", "200.00", batch=""),
            P("T3", "50.00", batch="", merchant="M999")]
    r = reconcile(*frames(rows, [K("BNK1", "B001", "295.50")]))
    assert r.fallback_assignments == {"T2": "B001"}
    assert r.manual_review == ["T3"]
    assert r.batch("B001").status == "MATCHED"


def test_duplicate_removal_happens_before_batch_comparison():
    rows = [P("T1", "100.00")]
    p, b = frames(rows, [K("BNK1", "B001", "98.50"), K("BNK1", "B001", "98.50")])
    r = reconcile(p, b)
    assert r.batch("B001").status == "MATCHED"           # would be UNEXPLAINED (-98.50) if compared first
    assert r.batch("B001").bank_credit == D("98.50")
    assert r.summary["total_bank_credits"] == D("98.50")
    assert r.summary["total_bank_credits_as_received"] == D("197.00")
    assert r.summary["unexplained"] == D("0.00")


def test_near_duplicates_are_out_of_scope():
    p, b = frames([P("T1", "100.00")],
                  [K("BNK1", "B001", "98.50", utr="UTR-A"), K("BNK1", "B001", "98.50", utr="UTR-B")])
    r = reconcile(p, b)
    assert not r.exceptions_of("DUPLICATE_BANK_ENTRY")   # same id but different UTR: kept
    assert r.batch("B001").status == "UNEXPLAINED"


def test_pending_vs_potentially_missing_depends_on_deadline():
    not_due = reconcile(*frames([P("T1", "100.00", expected="2026-11-03", tdate="2026-10-30")], []))
    assert not_due.batch("B001").status == "PENDING"
    assert not not_due.exceptions_of("POTENTIALLY_MISSING")
    assert not_due.equation["unexplained"] == D("0.00")
    overdue = reconcile(*frames([P("T1", "100.00", expected="2026-10-29")], []))
    assert overdue.batch("B001").status == "POTENTIALLY_MISSING"
    assert overdue.equation["unexplained"] == D("98.50") and overdue.equation["holds"]


def test_bank_credit_without_platform_batch_is_unexplained():
    r = reconcile(*frames([], [K("BNK9", "B777", "500.00")]))
    assert [e.gap_type for e in r.exceptions] == ["UNMATCHED_BANK_CREDIT"]
    assert r.equation["unexplained"] == D("-500.00") and r.equation["holds"]


@pytest.mark.parametrize("kind", ["no_columns", "header_only"])
def test_empty_input(kind):
    if kind == "no_columns":
        r = reconcile(pd.DataFrame(), pd.DataFrame())
    else:
        r = reconcile(*frames([], []))
    assert r.batches == [] and r.exceptions == []
    assert all(v == D("0.00") for v in r.summary.values())
    assert r.equation["holds"] and r.equation["unexplained"] == D("0.00")
    assert r.exceptions_df().empty and r.batches_df().empty
    assert "0.00 = 0.00 + 0.00" in r.equation_text()


def test_floats_are_rejected():
    p, b = frames([P(1, 100.0)], [])
    with pytest.raises(TypeError):
        reconcile(p, b)


def test_refund_reduces_batch_net_and_no_fee_reversal():
    rows = [P("T1", "1000.00"), P("R1", "200.00", ttype="REFUND", original="T1")]
    r = reconcile(*frames(rows, [K("BNK1", "B001", "785.00")]))   # 1000 - 200 - 15.00 fee (not 12.00)
    assert r.batch("B001").status == "MATCHED"
    assert r.batch("B001").expected_net == D("785.00")


# =============================================================== generator / hygiene checks
def test_generated_data_shape(platform, bank):
    assert len(platform) == 250
    assert len(bank) == 19 and bank["bank_transaction_id"].nunique() == 18
    assert platform["settlement_batch_id"].replace("", pd.NA).nunique() == 18
    assert 3 <= (platform["settlement_batch_id"] == "").sum() <= 5
    assert set(platform["currency"]) == set(bank["currency"]) == {"INR"}
    assert all(re.fullmatch(r"\d+\.\d\d", a) for a in platform["amount"])    # strings with 2 decimals


def test_generator_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate_data.generate(a, seed=42)
    generate_data.generate(b, seed=42)
    for name in ("platform_transactions.csv", "bank_transactions.csv", "ground_truth.csv"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_reconciliation_code_never_touches_ground_truth():
    for name in ("reconcile.py", "app.py"):
        assert "ground_truth" not in (ROOT / name).read_text(encoding="utf-8"), name


def test_streamlit_app_renders(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    generate_data.generate(tmp_path, seed=42)
    monkeypatch.setenv("RECON_DATA_DIR", str(tmp_path))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    values = {m.label: m.value for m in at.metric}
    assert values["Unexplained difference"] == "0.00"
    assert values["Sum of explained gaps"] == "3,644.51"
    assert len(at.dataframe) >= 1
    at.sidebar.checkbox[0].check().run()                 # include November credits
    assert not at.exception
    assert {m.label: m.value for m in at.metric}["Total difference (expected - bank)"] == "0.01"
