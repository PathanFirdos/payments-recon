"""Payments reconciliation: platform transactions vs bank settlement credits.

Sign convention everywhere: difference = EXPECTED (platform) - BANK.
  * positive  -> the bank paid less than expected (e.g. a late settlement)
  * negative  -> the bank paid more than expected (e.g. a duplicate bank credit)

This module never reads the test answer-key file. It works only from the two input frames.
All money is decimal.Decimal; floats are rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

import pandas as pd

FEE_RATE = Decimal("0.015")
CENT = Decimal("0.01")
ZERO = Decimal("0.00")
PERIOD_START = date(2026, 10, 1)
PERIOD_END = date(2026, 10, 31)
CURRENCY = "INR"

PLATFORM_COLUMNS = [
    "transaction_id", "customer_id", "merchant_id", "transaction_date", "transaction_type",
    "amount", "currency", "status", "original_transaction_id", "settlement_batch_id",
    "expected_settlement_date", "gateway_fee_amount",
]
BANK_COLUMNS = [
    "bank_transaction_id", "utr_number", "settlement_batch_id", "merchant_id", "bank_date",
    "credit_amount", "currency", "transaction_type",
]

# batch statuses
MATCHED = "MATCHED"
ROUNDING = "ROUNDING"
PENDING = "PENDING"
POTENTIALLY_MISSING = "POTENTIALLY_MISSING"
UNEXPLAINED = "UNEXPLAINED"
# exception types that are not batch statuses
LATE_SETTLEMENT = "LATE_SETTLEMENT"
DUPLICATE_BANK_ENTRY = "DUPLICATE_BANK_ENTRY"
ORPHAN_REFUND = "ORPHAN_REFUND"
MANUAL_REVIEW = "MANUAL_REVIEW"
UNMATCHED_BANK_CREDIT = "UNMATCHED_BANK_CREDIT"

_TYPE_ORDER = [LATE_SETTLEMENT, ROUNDING, DUPLICATE_BANK_ENTRY, ORPHAN_REFUND,
               MANUAL_REVIEW, POTENTIALLY_MISSING, UNMATCHED_BANK_CREDIT, UNEXPLAINED]


# --------------------------------------------------------------------------- money & parsing
def _dec(value, name: str = "amount") -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (float, bool)):
        raise TypeError(f"{name}: money must be str/Decimal, never float (got {value!r})")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            raise ValueError(f"{name} is blank")
        try:
            return Decimal(s)
        except InvalidOperation:
            raise ValueError(f"{name}: not a decimal number: {value!r}") from None
    raise TypeError(f"{name}: unsupported type {type(value).__name__}")


def q2(value) -> Decimal:
    return _dec(value).quantize(CENT, rounding=ROUND_HALF_UP)


def gateway_fee(amount: Decimal) -> Decimal:
    """1.5% fee rounded to 2 decimals, ROUND_HALF_UP."""
    return q2(_dec(amount) * FEE_RATE)


def _text(value) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value).strip()


def _date(value, name: str) -> date | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"{name}: expected YYYY-MM-DD, got {value!r}") from None


def _sum(values) -> Decimal:
    return sum(values, ZERO)


def _fmt(x: Decimal) -> str:
    return f"{x:.2f}"


# --------------------------------------------------------------------------- input rows
@dataclass
class _Txn:
    tid: str
    merchant: str
    tdate: date
    ttype: str
    amount: Decimal
    status: str
    original: str
    batch: str
    expected: date | None


@dataclass
class _Bank:
    bank_id: str
    utr: str
    batch_id: str
    bank_date: date
    credit: Decimal


def _frame(df, columns, label):
    if df is None:
        df = pd.DataFrame(columns=columns)
    missing = [c for c in columns if c not in df.columns]
    if missing and len(df):
        raise ValueError(f"{label}: missing columns {missing}")
    df = df.copy()
    for c in missing:
        df[c] = ""
    return df[columns]


def _prepare_platform(df) -> list[_Txn]:
    out = []
    for rec in _frame(df, PLATFORM_COLUMNS, "platform").to_dict("records"):
        ttype = _text(rec["transaction_type"]).upper()
        status = _text(rec["status"]).upper()
        if ttype not in ("PAYMENT", "REFUND"):
            raise ValueError(f"unknown transaction_type {ttype!r} on {rec['transaction_id']}")
        if status not in ("SUCCESS", "FAILED"):
            raise ValueError(f"unknown status {status!r} on {rec['transaction_id']}")
        if _text(rec["currency"]).upper() != CURRENCY:
            raise ValueError(f"only {CURRENCY} is supported (transaction {rec['transaction_id']})")
        tdate = _date(rec["transaction_date"], "transaction_date")
        if tdate is None:
            raise ValueError(f"transaction_date is blank on {rec['transaction_id']}")
        out.append(_Txn(
            tid=_text(rec["transaction_id"]), merchant=_text(rec["merchant_id"]), tdate=tdate,
            ttype=ttype, amount=_dec(rec["amount"], f"amount of {rec['transaction_id']}"),
            status=status, original=_text(rec["original_transaction_id"]),
            batch=_text(rec["settlement_batch_id"]),
            expected=_date(rec["expected_settlement_date"], "expected_settlement_date"),
        ))
    return out


def _prepare_bank(df) -> list[_Bank]:
    out = []
    for rec in _frame(df, BANK_COLUMNS, "bank").to_dict("records"):
        if _text(rec["currency"]).upper() != CURRENCY:
            raise ValueError(f"only {CURRENCY} is supported (bank row {rec['bank_transaction_id']})")
        if _text(rec["transaction_type"]).upper() != "SETTLEMENT_CREDIT":
            raise ValueError(f"unknown bank transaction_type on {rec['bank_transaction_id']}")
        bdate = _date(rec["bank_date"], "bank_date")
        if bdate is None:
            raise ValueError(f"bank_date is blank on {rec['bank_transaction_id']}")
        out.append(_Bank(
            bank_id=_text(rec["bank_transaction_id"]), utr=_text(rec["utr_number"]),
            batch_id=_text(rec["settlement_batch_id"]), bank_date=bdate,
            credit=_dec(rec["credit_amount"], f"credit_amount of {rec['bank_transaction_id']}"),
        ))
    return out


def _dedupe_bank(rows: list[_Bank]):
    """Duplicate = same bank_transaction_id AND same utr_number. First row is kept.
    Near-duplicates (only one of the two matches) are out of scope and kept."""
    seen, unique, dups = set(), [], []
    for r in rows:
        key = (r.bank_id, r.utr)
        if key in seen:
            dups.append(r)
        else:
            seen.add(key)
            unique.append(r)
    return unique, dups


# --------------------------------------------------------------------------- results
@dataclass
class BatchResult:
    batch_id: str
    merchant_id: str
    expected_date: date | None
    payment_ids: list
    refund_ids: list
    gross: Decimal
    refunds: Decimal
    fee_per_txn: Decimal
    fee_on_total: Decimal
    expected_net: Decimal
    bank_credit: Decimal
    bank_ids: list
    difference: Decimal
    status: str
    note: str = ""


@dataclass
class ExceptionRow:
    gap_type: str
    affected_ids: list
    amount: Decimal            # raw / gross amount, as shown in the exception table
    reason: str
    effect_dedup: Decimal      # signed effect on (expected - bank after duplicate removal)
    effect_statement: Decimal  # signed effect on (expected - bank as received, duplicates included)
    explained: bool
    batch_id: str = ""


@dataclass
class ReconResult:
    period_start: date
    period_end: date
    include_post_period_credits: bool
    batches: list
    exceptions: list
    summary: dict
    equation: dict
    fallback_assignments: dict = field(default_factory=dict)
    excluded_failed: list = field(default_factory=list)
    orphan_refunds: list = field(default_factory=list)
    duplicates_removed: list = field(default_factory=list)
    manual_review: list = field(default_factory=list)
    post_period_credits: list = field(default_factory=list)
    ignored: dict = field(default_factory=dict)

    def batch(self, batch_id: str) -> BatchResult:
        for b in self.batches:
            if b.batch_id == batch_id:
                return b
        raise KeyError(batch_id)

    def exceptions_of(self, gap_type: str) -> list:
        return [e for e in self.exceptions if e.gap_type == gap_type]

    # ---- display helpers (plain strings; Decimal stays authoritative on the dataclasses)
    def exceptions_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"gap_type": e.gap_type, "affected_ids": "; ".join(e.affected_ids),
              "amount": _fmt(e.amount), "reason": e.reason} for e in self.exceptions],
            columns=["gap_type", "affected_ids", "amount", "reason"])

    def batches_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "batch_id": b.batch_id, "merchant_id": b.merchant_id,
            "expected_date": b.expected_date.isoformat() if b.expected_date else "",
            "gross_payments": _fmt(b.gross), "valid_refunds": _fmt(b.refunds),
            "fee": _fmt(b.fee_per_txn), "expected_net": _fmt(b.expected_net),
            "bank_credit": _fmt(b.bank_credit), "difference": _fmt(b.difference),
            "status": b.status,
        } for b in self.batches])

    def summary_df(self) -> pd.DataFrame:
        s = self.summary
        rows = [
            ("Total platform payments (successful, gross)", s["total_platform_payments"]),
            ("Total expected settlement (net of fees and valid refunds)", s["total_expected_settlement"]),
            ("Total bank credits (after removing duplicates)", s["total_bank_credits"]),
            ("Total difference (expected - bank)", s["total_difference"]),
            ("Sum of explained gaps", s["net_explained"]),
            ("Unexplained difference", s["unexplained"]),
        ]
        return pd.DataFrame([{"item": k, "amount": _fmt(v)} for k, v in rows])

    def equation_text(self) -> str:
        e = self.equation
        c = e["components"]
        lines = [
            "Total difference = Net explained difference + Unexplained difference",
            f"{_fmt(e['total_difference'])} = {_fmt(e['net_explained'])} + {_fmt(e['unexplained'])}",
            "",
            "Net explained = "
            f"late settlement (net of fee) {_fmt(c[LATE_SETTLEMENT])}"
            f" + rounding {_fmt(c[ROUNDING])}"
            f" + duplicate bank entry {_fmt(c[DUPLICATE_BANK_ENTRY])} (already removed from bank total)"
            f" + orphan refund {_fmt(c[ORPHAN_REFUND])}",
            "",
            "Statement view (bank credits exactly as received, duplicate still counted):",
            f"{_fmt(e['statement_total_difference'])} = {_fmt(e['statement_net_explained'])}"
            f" + {_fmt(e['unexplained'])}",
            "Net explained = "
            f"late settlement {_fmt(e['statement_components'][LATE_SETTLEMENT])}"
            f" + rounding {_fmt(e['statement_components'][ROUNDING])}"
            f" + duplicate bank entry {_fmt(e['statement_components'][DUPLICATE_BANK_ENTRY])}"
            f" + orphan refund {_fmt(e['statement_components'][ORPHAN_REFUND])}",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- engine
def reconcile(platform, bank, period_start: date = PERIOD_START, period_end: date = PERIOD_END,
              include_post_period_credits: bool = False) -> ReconResult:
    """Reconcile platform transactions against bank credits.

    include_post_period_credits=False -> bank credits dated after period_end are not part of the
    totals (the cut-off view); late settlements show as PENDING / in transit.
    include_post_period_credits=True  -> November credits are included and late batches match.
    """
    txns_all = _prepare_platform(platform)
    bank_all = _prepare_bank(bank)

    # 1. Duplicate bank rows go FIRST, before any batch comparison or total.
    unique_bank, dup_bank = _dedupe_bank(bank_all)

    def bank_in_scope(d: date) -> bool:
        return d >= period_start and (include_post_period_credits or d <= period_end)

    bank_scope = [b for b in unique_bank if bank_in_scope(b.bank_date)]
    dup_scope = [b for b in dup_bank if bank_in_scope(b.bank_date)]
    post_period = [] if include_post_period_credits else [b for b in unique_bank if b.bank_date > period_end]
    ignored = {
        "platform_rows_outside_period": sum(1 for t in txns_all if not period_start <= t.tdate <= period_end),
        "bank_rows_before_period": sum(1 for b in unique_bank if b.bank_date < period_start),
    }

    # 2. Platform rows in the reporting period.
    payment_ids_all = {t.tid for t in txns_all if t.ttype == "PAYMENT"}
    txns = [t for t in txns_all if period_start <= t.tdate <= period_end]

    # 3. Recover blank settlement_batch_id by merchant_id + expected_settlement_date. Never guess.
    known: dict[str, set] = {}
    for t in txns:
        if t.batch:
            known.setdefault(t.batch, set()).add((t.merchant, t.expected))
    fallback: dict[str, str] = {}
    manual: list[tuple[_Txn, list]] = []
    for t in txns:
        if t.batch or t.status != "SUCCESS":
            continue
        cands = sorted(b for b, keys in known.items()
                       if t.merchant and t.expected and (t.merchant, t.expected) in keys)
        if len(cands) == 1:
            t.batch = cands[0]
            fallback[t.tid] = cands[0]
        else:
            manual.append((t, cands))

    # 4. Orphan refunds, failed rows.
    orphans = [t for t in txns if t.ttype == "REFUND" and t.status == "SUCCESS"
               and (not t.original or t.original not in payment_ids_all)]
    orphan_ids = {t.tid for t in orphans}
    failed = [t for t in txns if t.status == "FAILED"]

    by_batch: dict[str, list[_Txn]] = {}
    for t in txns:
        if t.batch and t.status == "SUCCESS" and t.tid not in orphan_ids:
            by_batch.setdefault(t.batch, []).append(t)
    bank_by_batch: dict[str, list[_Bank]] = {}
    for b in bank_scope:
        bank_by_batch.setdefault(b.batch_id, []).append(b)
    post_by_batch: dict[str, list[_Bank]] = {}
    for b in post_period:
        post_by_batch.setdefault(b.batch_id, []).append(b)

    # 5. Batch comparison.
    batches: list[BatchResult] = []
    exceptions: list[ExceptionRow] = []
    for bid in sorted(by_batch):
        rows = by_batch[bid]
        pays = [t for t in rows if t.ttype == "PAYMENT"]
        refs = [t for t in rows if t.ttype == "REFUND"]
        gross = _sum(t.amount for t in pays)
        refunds = _sum(t.amount for t in refs)
        fee_txn = _sum(gateway_fee(t.amount) for t in pays)
        fee_batch = gateway_fee(gross)
        expected_net = gross - refunds - fee_txn
        credits = bank_by_batch.get(bid, [])
        bank_amt = _sum(c.credit for c in credits)
        diff = expected_net - bank_amt
        exp_dates = [t.expected for t in rows if t.expected]
        exp_date = max(exp_dates) if exp_dates else None
        pay_ids = [t.tid for t in pays]
        bank_ids = [c.bank_id for c in credits]
        note = ""
        status = MATCHED

        if not credits:
            post = post_by_batch.get(bid, [])
            if expected_net == ZERO:
                status = MATCHED
            elif exp_date is not None and exp_date > period_end:
                status = PENDING
                where = (f"bank credit {', '.join(p.bank_id for p in post)} of "
                         f"{_fmt(_sum(p.credit for p in post))} arrives "
                         f"{', '.join(sorted({p.bank_date.isoformat() for p in post}))}"
                         if post else "no bank credit seen yet")
                note = (f"Pending / in transit: expected settlement {exp_date.isoformat()} is after the "
                        f"{period_end.isoformat()} cut-off; {where}. Not missing.")
                exceptions.append(ExceptionRow(
                    LATE_SETTLEMENT, pay_ids + [bid] + [p.bank_id for p in post], gross, note,
                    expected_net, expected_net, True, bid))
            else:
                status = POTENTIALLY_MISSING
                extra = (f" A credit dated after the cut-off exists ({', '.join(p.bank_id for p in post)})."
                         if post else "")
                note = (f"No bank credit although expected settlement "
                        f"{exp_date.isoformat() if exp_date else 'date unknown'} has passed.{extra}")
                exceptions.append(ExceptionRow(
                    POTENTIALLY_MISSING, [bid] + pay_ids, expected_net, note,
                    expected_net, expected_net, False, bid))
        elif diff == ZERO:
            status = MATCHED
        else:
            proof = (fee_batch - fee_txn)
            if abs(diff) == CENT and proof == diff:
                status = ROUNDING
                note = (f"Fee rounding: per-transaction fees total {_fmt(fee_txn)}, one fee on the batch "
                        f"total is {_fmt(fee_batch)}; they differ by {_fmt(proof)}, which is the "
                        f"{_fmt(diff)} difference.")
                exceptions.append(ExceptionRow(
                    ROUNDING, pay_ids + [bid] + bank_ids, abs(diff), note, diff, diff, True, bid))
            else:
                status = UNEXPLAINED
                direction = "bank paid less than expected" if diff > 0 else "bank paid more than expected"
                if abs(diff) == CENT:
                    note = (f"0.01 difference without fee-rounding proof (per-transaction fees "
                            f"{_fmt(fee_txn)} vs batch fee {_fmt(fee_batch)}); {direction}.")
                else:
                    note = f"Difference of {_fmt(abs(diff))}: {direction}; no known cause."
                exceptions.append(ExceptionRow(
                    UNEXPLAINED, [bid] + bank_ids, abs(diff), note, diff, diff, False, bid))

        batches.append(BatchResult(
            batch_id=bid, merchant_id=rows[0].merchant, expected_date=exp_date, payment_ids=pay_ids,
            refund_ids=[t.tid for t in refs], gross=gross, refunds=refunds, fee_per_txn=fee_txn,
            fee_on_total=fee_batch, expected_net=expected_net, bank_credit=bank_amt, bank_ids=bank_ids,
            difference=diff, status=status, note=note))

    # 6. Bank credits that belong to no platform batch.
    for b in bank_scope:
        if b.batch_id not in by_batch:
            exceptions.append(ExceptionRow(
                UNMATCHED_BANK_CREDIT, [b.bank_id, b.batch_id or "(no batch)"], b.credit,
                f"Bank credit has no platform batch ({b.batch_id or 'blank'}); bank paid more than expected.",
                -b.credit, -b.credit, False, b.batch_id))

    # 7. Duplicate bank entries (already removed above).
    for d in dup_bank:
        in_scope = bank_in_scope(d.bank_date)
        exceptions.append(ExceptionRow(
            DUPLICATE_BANK_ENTRY, [d.bank_id, d.utr, d.batch_id], d.credit,
            f"Same bank_transaction_id and UTR posted twice; duplicate removed before comparison. "
            f"Bank total was overstated by {_fmt(d.credit)}.",
            ZERO, -d.credit if in_scope else ZERO, True, d.batch_id))

    # 8. Orphan refunds and manual-review rows (no effect on any bank comparison).
    for t in orphans:
        ref = t.original or "(blank)"
        exceptions.append(ExceptionRow(
            ORPHAN_REFUND, [t.tid, ref], t.amount,
            f"Refund {t.tid} refers to {ref}, which does not exist; excluded from batch calculations "
            f"(data-quality exception, not a bank discrepancy).", ZERO, ZERO, True, t.batch))
    for t, cands in manual:
        why = (f"{len(cands)} batches fit merchant {t.merchant} + expected date "
               f"{t.expected.isoformat() if t.expected else 'unknown'} ({', '.join(cands)})"
               if cands else "no batch fits merchant + expected settlement date")
        exceptions.append(ExceptionRow(
            MANUAL_REVIEW, [t.tid] + cands, t.amount,
            f"Blank settlement_batch_id: {why}. Not assigned (never guess); excluded from expected settlement.",
            ZERO, ZERO, False, ""))

    exceptions.sort(key=lambda e: (_TYPE_ORDER.index(e.gap_type), e.affected_ids))

    # 9. Totals and the equation.
    total_platform = _sum(t.amount for t in txns if t.ttype == "PAYMENT" and t.status == "SUCCESS")
    total_expected = _sum(b.expected_net for b in batches)
    total_bank = _sum(b.credit for b in bank_scope)
    total_bank_raw = total_bank + _sum(d.credit for d in dup_scope)
    total_difference = total_expected - total_bank
    statement_difference = total_expected - total_bank_raw

    explained = [e for e in exceptions if e.explained]
    unexplained_ex = [e for e in exceptions if not e.explained]
    net_explained = _sum(e.effect_dedup for e in explained)
    unexplained = _sum(e.effect_dedup for e in unexplained_ex)
    st_net_explained = _sum(e.effect_statement for e in explained)
    st_unexplained = _sum(e.effect_statement for e in unexplained_ex)

    def comp(attr):
        return {g: _sum(getattr(e, attr) for e in explained if e.gap_type == g)
                for g in (LATE_SETTLEMENT, ROUNDING, DUPLICATE_BANK_ENTRY, ORPHAN_REFUND)}

    equation = {
        "total_difference": total_difference, "net_explained": net_explained, "unexplained": unexplained,
        "holds": total_difference == net_explained + unexplained, "components": comp("effect_dedup"),
        "statement_total_difference": statement_difference, "statement_net_explained": st_net_explained,
        "statement_unexplained": st_unexplained,
        "statement_holds": statement_difference == st_net_explained + st_unexplained,
        "statement_components": comp("effect_statement"),
    }
    summary = {
        "total_platform_payments": total_platform, "total_expected_settlement": total_expected,
        "total_bank_credits_as_received": total_bank_raw, "duplicates_removed": total_bank_raw - total_bank,
        "total_bank_credits": total_bank, "total_difference": total_difference,
        "net_explained": net_explained, "unexplained": unexplained,
    }
    return ReconResult(
        period_start=period_start, period_end=period_end,
        include_post_period_credits=include_post_period_credits, batches=batches, exceptions=exceptions,
        summary=summary, equation=equation, fallback_assignments=fallback,
        excluded_failed=[{"transaction_id": t.tid, "type": t.ttype, "batch_id": t.batch,
                          "amount": t.amount} for t in failed],
        orphan_refunds=[t.tid for t in orphans], duplicates_removed=[d.bank_id for d in dup_bank],
        manual_review=[t.tid for t, _ in manual],
        post_period_credits=[{"bank_transaction_id": b.bank_id, "batch_id": b.batch_id,
                              "bank_date": b.bank_date.isoformat(), "credit_amount": b.credit}
                             for b in post_period],
        ignored=ignored)


# --------------------------------------------------------------------------- CSV helpers
def _read(path) -> pd.DataFrame:
    return pd.read_csv(Path(path), dtype=str, keep_default_na=False)


def load_platform(path) -> pd.DataFrame:
    return _read(path)


def load_bank(path) -> pd.DataFrame:
    return _read(path)


def reconcile_files(platform_csv, bank_csv, **kwargs) -> ReconResult:
    return reconcile(load_platform(platform_csv), load_bank(bank_csv), **kwargs)


if __name__ == "__main__":
    import sys

    data = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    res = reconcile_files(data / "platform_transactions.csv", data / "bank_transactions.csv")
    print(res.summary_df().to_string(index=False))
    print()
    print(res.equation_text())
    print()
    print(res.exceptions_df().to_string(index=False))
