#!/usr/bin/env python3
"""Generate the seeded October 2026 payments data set.

Writes three CSVs (seed 42, INR only):
  platform_transactions.csv  250 rows
  bank_transactions.csv       19 rows (18 settlement credits + 1 duplicate)
  ground_truth.csv            planted gaps + expected non-exception outcomes

ground_truth.csv is for the TEST SUITE ONLY. reconcile.py and app.py never read it.

This generator deliberately re-implements the fee and business-day rules on its own
(instead of importing reconcile.py) so the tests compare two independent implementations.
Bank credits are produced the way a bank would: ONE fee computed on the batch total.
Platform expectations use the per-transaction fee. For normal batches every amount is a
multiple of INR 2.00, so both methods agree exactly; only the planted batch B007 differs.
"""
from __future__ import annotations

import argparse
import csv
import random
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

SEED = 42
CENT = Decimal("0.01")
FEE_RATE = Decimal("0.015")

PLATFORM_COLUMNS = [
    "transaction_id", "customer_id", "merchant_id", "transaction_date", "transaction_type",
    "amount", "currency", "status", "original_transaction_id", "settlement_batch_id",
    "expected_settlement_date", "gateway_fee_amount",
]
BANK_COLUMNS = [
    "bank_transaction_id", "utr_number", "settlement_batch_id", "merchant_id", "bank_date",
    "credit_amount", "currency", "transaction_type",
]
GROUND_TRUTH_COLUMNS = ["id", "gap_type", "affected_ids", "expected_amount"]


def _fee(amount: Decimal) -> Decimal:
    return (amount * FEE_RATE).quantize(CENT, rounding=ROUND_HALF_UP)


def _m(value: Decimal) -> str:
    return f"{value.quantize(CENT):.2f}"


def _add_business_days(start: date, n: int) -> date:
    d = start
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def generate(out_dir: str | Path, seed: int = SEED) -> dict:
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ batches
    oct1 = date(2026, 10, 1)
    bdays = [oct1 + timedelta(days=i) for i in range(27)]  # Oct 1..Oct 27
    bdays = [d for d in bdays if d.weekday() < 5]
    merchants = ["M001", "M002", "M003", "M004"]
    pairs = sorted((d, m) for d in bdays for m in merchants)
    chosen = sorted(rng.sample(pairs, 16))

    batches: dict[str, dict] = {}
    for i, (d, m) in enumerate(chosen, start=1):
        batches[f"B{i:03d}"] = dict(
            merchant=m, txn_date=d, expected=_add_business_days(d, 2),
            bank_date=_add_business_days(d, rng.choice([1, 2])),
        )
    # Late settlements: recorded Fri Oct 30 / Sat Oct 31, credited in November.
    batches["B017"] = dict(merchant="M002", txn_date=date(2026, 10, 30),
                           expected=date(2026, 11, 3), bank_date=date(2026, 11, 3))
    batches["B018"] = dict(merchant="M003", txn_date=date(2026, 10, 31),
                           expected=date(2026, 11, 3), bank_date=date(2026, 11, 2))
    assert _add_business_days(date(2026, 10, 30), 2) == date(2026, 11, 3)
    assert _add_business_days(date(2026, 10, 31), 2) == date(2026, 11, 3)

    # ------------------------------------------------------- payment -> batch map
    assignment: dict[int, str] = {}
    pos = 0
    for bid, n in zip([f"B{i:03d}" for i in range(1, 7)], [12, 12, 12, 12, 11, 11]):
        for _ in range(n):
            assignment[1001 + pos] = bid
            pos += 1
    assert 1001 + pos == 1071
    for n in (1071, 1072, 1073):
        assignment[n] = "B007"
    pos = 0
    for bid, n in zip([f"B{i:03d}" for i in range(8, 17)], [15] + [14] * 8):
        for _ in range(n):
            assignment[1074 + pos] = bid
            pos += 1
    assert 1074 + pos == 1201
    fallback_truth = {1201: "B004", 1202: "B009", 1203: "B013", 1204: "B015"}
    assignment.update(fallback_truth)
    normal_batches = [f"B{i:03d}" for i in range(1, 17) if i != 7]
    for n in range(1205, 1231):
        assignment[n] = rng.choice(normal_batches)

    amounts: dict[int, Decimal] = {}
    for n, bid in sorted(assignment.items()):
        if bid == "B007":
            amounts[n] = Decimal("1010.30")  # fee 15.1545 -> 15.15 each; batch fee 45.4635 -> 45.46
        else:
            amounts[n] = Decimal(rng.randrange(100, 3000, 2))  # even rupees: 1.5% is exact in paise

    # B012: gross 5000.00, one 675.00 refund -> net exactly 4250.00 (the duplicated credit)
    b12 = sorted(n for n, b in assignment.items() if b == "B012")
    while True:
        draw = [Decimal(rng.randrange(100, 400, 2)) for _ in b12[:-1]]
        last = Decimal(5000) - sum(draw)
        if 700 <= last <= 3000:
            break
    for n, a in zip(b12, draw + [last]):
        amounts[n] = a

    customers = {n: f"C{rng.randrange(1, 301):04d}" for n in assignment}

    # ------------------------------------------------------------- platform rows
    rows: list[dict] = []
    pay_by_batch: dict[str, list[dict]] = {b: [] for b in batches}

    def payment_row(tid, cust, bid, amount, status="SUCCESS", blank_batch=False):
        b = batches[bid]
        row = {
            "transaction_id": tid, "customer_id": cust, "merchant_id": b["merchant"],
            "transaction_date": b["txn_date"].isoformat(), "transaction_type": "PAYMENT",
            "amount": _m(amount), "currency": "INR", "status": status,
            "original_transaction_id": "",
            "settlement_batch_id": "" if blank_batch else bid,
            "expected_settlement_date": b["expected"].isoformat(),
            "gateway_fee_amount": _m(_fee(amount)) if status == "SUCCESS" else "0.00",
            "_amount": amount, "_batch": bid,
        }
        rows.append(row)
        pay_by_batch[bid].append(row)
        return row

    for n, bid in sorted(assignment.items()):
        payment_row(f"TXN{n}", customers[n], bid, amounts[n], blank_batch=n in fallback_truth)

    payment_row("TXN1249", "C0042", "B017", Decimal("1500.00"))
    payment_row("TXN1250", "C0043", "B018", Decimal("2200.00"))
    failed = {"TXN1301": ("B003", Decimal("1200.00")), "TXN1302": ("B009", Decimal("850.00")),
              "TXN1303": ("B014", Decimal("2400.00"))}
    for tid, (bid, amt) in failed.items():
        payment_row(tid, f"C{rng.randrange(1, 301):04d}", bid, amt, status="FAILED")

    # refunds: 14 valid (RFND3002..3015) + 1 orphan (RFND3001 -> TXN9999)
    refund_batches = ["B001", "B002", "B003", "B004", "B005", "B006", "B008", "B009", "B010",
                      "B011", "B012", "B013", "B015", "B016"]
    refunds_by_batch: dict[str, list[Decimal]] = {b: [] for b in batches}
    for k, bid in enumerate(refund_batches):
        rid = f"RFND{3002 + k}"
        pool = [r for r in pay_by_batch[bid]
                if r["status"] == "SUCCESS" and r["transaction_id"] not in {f"TXN{n}" for n in fallback_truth}]
        if bid == "B012":
            orig = max(pool, key=lambda r: r["_amount"])
            amt = Decimal(675)
        else:
            orig = rng.choice(pool)
            amt = Decimal(rng.randrange(50, int(min(orig["_amount"], 1000)) + 1))
        b = batches[bid]
        rows.append({
            "transaction_id": rid, "customer_id": orig["customer_id"], "merchant_id": b["merchant"],
            "transaction_date": b["txn_date"].isoformat(), "transaction_type": "REFUND",
            "amount": _m(amt), "currency": "INR", "status": "SUCCESS",
            "original_transaction_id": orig["transaction_id"], "settlement_batch_id": bid,
            "expected_settlement_date": b["expected"].isoformat(), "gateway_fee_amount": "0.00",
        })
        refunds_by_batch[bid].append(amt)
    b = batches["B010"]
    rows.append({
        "transaction_id": "RFND3001", "customer_id": "C0099", "merchant_id": b["merchant"],
        "transaction_date": b["txn_date"].isoformat(), "transaction_type": "REFUND",
        "amount": "300.00", "currency": "INR", "status": "SUCCESS",
        "original_transaction_id": "TXN9999", "settlement_batch_id": "B010",
        "expected_settlement_date": b["expected"].isoformat(), "gateway_fee_amount": "0.00",
    })  # orphan: NOT subtracted from the bank credit of B010

    rows.sort(key=lambda r: (r["transaction_date"], r["transaction_id"]))
    assert len(rows) == 250, len(rows)

    # ---------------------------------------------------------------- bank rows
    bank_rows: list[dict] = []
    fee_mismatch = []
    for i, bid in enumerate(sorted(batches), start=1):
        succ = [r["_amount"] for r in pay_by_batch[bid] if r["status"] == "SUCCESS"]
        gross = sum(succ, Decimal(0))
        if sum((_fee(a) for a in succ), Decimal(0)) != _fee(gross):
            fee_mismatch.append(bid)
        credit = gross - sum(refunds_by_batch[bid], Decimal(0)) - _fee(gross)  # bank: one fee on total
        bank_rows.append({
            "bank_transaction_id": f"BNK{5000 + i}", "utr_number": f"UTR{5000 + i}",
            "settlement_batch_id": bid, "merchant_id": batches[bid]["merchant"],
            "bank_date": batches[bid]["bank_date"].isoformat(), "credit_amount": _m(credit),
            "currency": "INR", "transaction_type": "SETTLEMENT_CREDIT",
        })
    assert fee_mismatch == ["B007"], fee_mismatch
    bank_rows.sort(key=lambda r: (r["bank_date"], r["bank_transaction_id"]))
    idx = next(i for i, r in enumerate(bank_rows) if r["bank_transaction_id"] == "BNK5012")
    assert bank_rows[idx]["credit_amount"] == "4250.00", bank_rows[idx]
    bank_rows.insert(idx + 1, dict(bank_rows[idx]))  # the duplicate posting
    assert len(bank_rows) == 19

    # -------------------------------------------------------------- ground truth
    gt = [
        ("GT001", "LATE_SETTLEMENT", "TXN1249;B017;BNK5017", "1500.00"),
        ("GT002", "LATE_SETTLEMENT", "TXN1250;B018;BNK5018", "2200.00"),
        ("GT003", "ROUNDING", "TXN1071;TXN1072;TXN1073;B007;BNK5007", "0.01"),
        ("GT004", "DUPLICATE_BANK_ENTRY", "BNK5012;UTR5012;B012", "4250.00"),
        ("GT005", "ORPHAN_REFUND", "RFND3001;TXN9999", "300.00"),
    ]
    k = 6
    for n, bid in sorted(fallback_truth.items()):
        gt.append((f"GT{k:03d}", "FALLBACK_MATCH", f"TXN{n};{bid}", _m(amounts[n])))
        k += 1
    for tid, (bid, amt) in failed.items():
        gt.append((f"GT{k:03d}", "EXCLUDED_FAILED", f"{tid};{bid}", _m(amt)))
        k += 1

    # -------------------------------------------------------------------- write
    def write(name, columns, records):
        with open(out_dir / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            w.writerows(records)

    write("platform_transactions.csv", PLATFORM_COLUMNS, rows)
    write("bank_transactions.csv", BANK_COLUMNS, bank_rows)
    write("ground_truth.csv", GROUND_TRUTH_COLUMNS,
          [dict(zip(GROUND_TRUTH_COLUMNS, r)) for r in gt])
    return {"platform_rows": len(rows), "bank_rows": len(bank_rows), "batches": len(batches),
            "blank_batch_rows": sum(1 for r in rows if r["settlement_batch_id"] == "")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data", help="output directory (default: data)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    print(generate(args.out, args.seed))
