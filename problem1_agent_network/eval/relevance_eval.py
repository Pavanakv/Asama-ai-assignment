"""Relevance-routing evaluation: precision / recall against hand-labelled posts, plus a threshold sweep.

Run from the project root:   python -m eval.relevance_eval

Labels were written BEFORE looking at scores. "expected" = agents a sensible business person would want
notified. Ambiguous posts were left out rather than labelled to suit the router. 15 posts is a smoke-level
benchmark, not a proof - the point is to have a repeatable number to improve against.
"""
from __future__ import annotations

import asyncio
import sys

from agents.world import build_world
from app.config import Settings

# (author, title, content, expected recipients)
LABELLED = [
    ("Swiggy", "Diwali delivery offer", "5% off all deliveries on Diwali", {"PhonePe", "HDFC Bank"}),
    ("Flipkart", "Big sale with card offers", "Our festive sale is live with an extra 10% discount on credit card and EMI payments", {"HDFC Bank", "PhonePe"}),
    ("PhonePe", "UPI cashback on food orders", "Flat cashback when customers pay for food delivery and grocery orders with UPI", {"Swiggy", "Flipkart"}),
    ("HDFC Bank", "Two-wheeler loan rates cut", "We are cutting loan interest rates for two-wheeler and motorcycle buyers this festive season", {"Bajaj Auto"}),
    ("Bajaj Auto", "New electric scooter launch", "We are launching a new electric scooter and looking for financing and EMI options for buyers at our dealerships", {"HDFC Bank"}),
    ("Flipkart", "Office cafeteria update", "The cafeteria is now serving masala chai and filter coffee from Monday", set()),
    ("Swiggy", "Looking for delivery partners", "Swiggy is looking for logistics and last-mile delivery partners to expand dark store fulfilment", {"Flipkart"}),
    ("PhonePe", "Offline merchant onboarding", "We are onboarding offline merchants and restaurants for UPI payments and want partner platforms to collaborate", {"Swiggy", "Flipkart"}),
    ("Bajaj Auto", "Plant maintenance shutdown", "The Chakan plant will be closed for annual maintenance for two days in March", set()),
    ("Swiggy", "Engineering hackathon winners", "Congratulations to the winners of our internal engineering hackathon", set()),
    ("HDFC Bank", "Card EMI on electronics", "No-cost EMI on credit cards for electronics purchases during the festive shopping sale", {"Flipkart"}),
    ("Flipkart", "Same-day delivery expansion", "We are expanding same-day delivery and logistics to 20 new cities", {"Swiggy"}),
    ("Bajaj Auto", "Test ride festival", "Test ride festival for motorcycles with cashback and card offers on bike bookings during the festive season", {"PhonePe", "HDFC Bank"}),
    ("PhonePe", "Company picnic", "Our annual team picnic is on Saturday at Cubbon Park", set()),
    ("HDFC Bank", "Branch timings", "Branch timings change to 10am-4pm on public holidays", set()),
]


async def run_once(threshold: float) -> tuple:
    w = await build_world(Settings(threshold=threshold))
    rows, tp, fp, fn = [], 0, 0, 0
    for author, title, content, expected in LABELLED:
        out = await w.post(author, title, content)
        got = {n["name"] for n in out["routing"]["notified"]} if out.get("routing") else set()
        tp += len(got & expected)
        fp += len(got - expected)
        fn += len(expected - got)
        rows.append((author, title, expected, got))
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1, rows, (tp, fp, fn)


async def main() -> None:
    default = Settings().threshold
    p, r, f1, rows, (tp, fp, fn) = await run_once(default)
    print(f"threshold={default}  precision={p:.2f}  recall={r:.2f}  F1={f1:.2f}   (TP={tp} FP={fp} FN={fn}, {len(LABELLED)} posts)\n")
    for author, title, expected, got in rows:
        flag = "ok " if got == expected else "ERR"
        miss, extra = sorted(expected - got), sorted(got - expected)
        note = (f"  missed: {miss}" if miss else "") + (f"  extra: {extra}" if extra else "")
        print(f"  {flag} {author:<10} {title:<32}{note}")
    print("\nthreshold sweep:")
    print("  thr    precision  recall   F1")
    for t in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60):
        p, r, f1, *_ = await run_once(t)
        print(f"  {t:.2f}   {p:6.2f}    {r:6.2f}  {f1:5.2f}{'   <- default' if abs(t - default) < 1e-9 else ''}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
