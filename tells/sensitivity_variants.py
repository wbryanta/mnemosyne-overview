#!/usr/bin/env python3
"""Aggregation sensitivity for the folk-tells combined score.

The published headline — the twelve tells combined score AUC 0.506, a
coin flip — pools the tells one particular way: per-tell z-scores against
the human-window mean/sd, summed in the folk direction (AI-high) over all
twelve. That is the honest test of the folk claim, because the claim on
offer is that *all of these* tells indicate AI. But it is one of several
defensible ways to pool twelve counters, and the choice moves the number
a lot. Rather than leave that for someone else to discover, this script
recomputes the combined AUC under every subset and pooling variant we
could think of, hostile ones included, from the same bundled rows that
`score_tells.py --reproduce` verifies.

What the spread shows: subsets chosen *with label knowledge* — keeping
only the tells that ran in the folk direction on this data — score
0.67-0.73, and an accuser in the wild does not have that oracle; they
have the circulating list, which is the first row. Two variants below
are marked as oracle-free because they could have been picked in
advance and still beat chance: the em dash alone, and the three tells
this project's own repository description used to name. They are here
because they are the honest counter-argument to the headline. What they
support is that the list is a coin flip and its best member is a weak,
model-specific signal — the em dash's cluster CI crosses chance, and
its per-model medians run 0.00-7.01. Both are reported per-tell in
README.md.

  python3 sensitivity_variants.py
      Print the variant table.

  python3 sensitivity_variants.py --reproduce
      Recompute every variant and verify it against the values published
      in ../README.md and README.md. PASS/FAIL per variant, nonzero exit
      on any mismatch (tolerance 1e-9).

Pure Python standard library (3.9+), like `score_tells.py`, whose
counters and statistics this reuses.

Two construct-validity variants are NOT here because they need the
source texts (copyrighted novels, not redistributed) rather than the
bundled rates: the strict reframing-only "not X, but Y" counter and the
`-est` blocklist leak. Both are measured and reported in README.md, with
the receipt scripts named there.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import score_tells as st  # noqa: E402

DATA_FILE = st.DATA_FILE

# The four tells that ran materially human-high on this data. Dropping
# them is the "forward subset" move: it requires knowing the labels.
BACKWARDS_TELLS = ["exclamation", "lets_opener", "superlative", "hedges"]
FORWARD_TELLS = [t for t in st.TELL_IDS if t not in BACKWARDS_TELLS]
HEADLINE_TELLS = ["em_dash", "not_x_but_y", "tricolon"]
BEST_PAIR = ["em_dash", "not_x_but_y"]
# The three tells the project's own GitHub description used to name.
NAMED_TRIPLE = ["em_dash", "not_x_but_y", "delve_leverage"]


# ---------------------------------------------------------------------------
# Pooling variants
# ---------------------------------------------------------------------------

def z_sum(human_rows: List[dict], ai_rows: List[dict], tells: Sequence[str],
          anchor: str = "human") -> Tuple[List[float], List[float]]:
    """Sum of per-tell z-scores. anchor='human' standardizes against the
    human windows (the published choice: the human side is the reference
    population an accusation is made against); anchor='pooled' uses both
    sides, which lets the AI samples inflate their own denominators."""
    base = human_rows if anchor == "human" else human_rows + ai_rows
    mu = {t: st._mean([r[t] for r in base]) for t in tells}
    sd = {t: st._std_ddof1([r[t] for r in base]) or 1.0 for t in tells}
    score = lambda rows: [  # noqa: E731
        math.fsum((r[t] - mu[t]) / sd[t] for t in tells) for r in rows]
    return score(human_rows), score(ai_rows)


def rank_sum(human_rows: List[dict], ai_rows: List[dict],
             tells: Sequence[str]) -> Tuple[List[float], List[float]]:
    """Sum of within-tell midranks over the pooled sample — scale-free, so
    a single heavy-tailed tell cannot dominate the sum."""
    rows = human_rows + ai_rows
    ranks: Dict[str, List[float]] = {}
    for t in tells:
        values = [r[t] for r in rows]
        order = sorted(range(len(values)), key=values.__getitem__)
        rk = [0.0] * len(values)
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
                j += 1
            midrank = (i + j + 2) / 2
            for k in range(i, j + 1):
                rk[order[k]] = midrank
            i = j + 1
        ranks[t] = rk
    total = [math.fsum(ranks[t][i] for t in tells) for i in range(len(rows))]
    return total[:len(human_rows)], total[len(human_rows):]


def log1p_rows(rows: List[dict]) -> List[dict]:
    """log1p the rates before pooling — the usual fix for count data whose
    variance tracks its mean."""
    return [{t: math.log1p(r[t]) for t in st.TELL_IDS} for r in rows]


def any_tell_over_p95(human_rows: List[dict], ai_rows: List[dict],
                      tells: Sequence[str]) -> Tuple[float, float]:
    """The rule an actual accuser applies: flag a document if ANY of the
    tells exceeds the human 95th percentile. Returns (ai_flagged_share,
    human_flagged_share)."""
    thr = {t: st._percentile([r[t] for r in human_rows], 95) for t in tells}
    flagged = lambda rows: sum(  # noqa: E731
        1 for r in rows if any(r[t] > thr[t] for t in tells)) / len(rows)
    return flagged(ai_rows), flagged(human_rows)


# ---------------------------------------------------------------------------
# Published values (README.md and ../README.md). --reproduce checks these.
# ---------------------------------------------------------------------------

# (key, label, published AUC)
PUBLISHED_AUC: List[Tuple[str, str, float]] = [
    ("published", "all 12, z-sum, human-anchored  [PUBLISHED]",
     0.5058205128205128),
    ("forward8", "8 forward tells (drop the 4 backwards)   *oracle*",
     0.6656955128205128),
    ("headline3", "3 headline tells: em dash/not-X-but-Y/tricolon  *oracle*",
     0.7056185897435898),
    ("pair2", "best pair: em dash + not-X-but-Y   *oracle*",
     0.7343237179487180),
    ("named3", "named triple: em dash/not-X-but-Y/delve",
     0.7275192307692308),
    ("em_dash_alone", "em dash alone (best single tell)",
     0.6803750000000000),
    ("pooled12", "all 12, z-sum, pooled-anchored",
     0.4510192307692308),
    ("ranksum12", "all 12, rank-sum pooling",
     0.4417019230769231),
    ("ranksum8", "8 forward tells, rank-sum pooling   *oracle*",
     0.6811378205128205),
    ("log1p12", "all 12, log1p rates, z-sum",
     0.4571666666666667),
    ("log1p8", "8 forward tells, log1p rates, z-sum   *oracle*",
     0.6549006410256410),
]

# (key, label, published AI flagged share, published human flagged share)
PUBLISHED_ANY_RULE: List[Tuple[str, str, float, float]] = [
    ("any12", "all 12 tells", 0.5275, 0.34871794871794873),
    ("any8", "8 forward tells", 0.5175, 0.23846153846153847),
    ("any3", "3 headline tells", 0.4425, 0.1282051282051282),
]


def compute(data: dict) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    """Every variant, from the bundled per-window/per-sample rate rows."""
    human_rows = data["human_window_rows"]
    ai_rows = data["ai_sample_rows"]
    human_log = log1p_rows(human_rows)
    ai_log = log1p_rows(ai_rows)

    def auc_of(pair: Tuple[List[float], List[float]]) -> float:
        human, ai = pair
        return st.mann_whitney_auc(ai, human)

    auc = {
        "published": auc_of(z_sum(human_rows, ai_rows, st.TELL_IDS)),
        "forward8": auc_of(z_sum(human_rows, ai_rows, FORWARD_TELLS)),
        "headline3": auc_of(z_sum(human_rows, ai_rows, HEADLINE_TELLS)),
        "pair2": auc_of(z_sum(human_rows, ai_rows, BEST_PAIR)),
        "named3": auc_of(z_sum(human_rows, ai_rows, NAMED_TRIPLE)),
        "em_dash_alone": st.mann_whitney_auc(
            [r["em_dash"] for r in ai_rows], [r["em_dash"] for r in human_rows]),
        "pooled12": auc_of(z_sum(human_rows, ai_rows, st.TELL_IDS, "pooled")),
        "ranksum12": auc_of(rank_sum(human_rows, ai_rows, st.TELL_IDS)),
        "ranksum8": auc_of(rank_sum(human_rows, ai_rows, FORWARD_TELLS)),
        "log1p12": auc_of(z_sum(human_log, ai_log, st.TELL_IDS)),
        "log1p8": auc_of(z_sum(human_log, ai_log, FORWARD_TELLS)),
    }
    any_rule = {
        "any12": any_tell_over_p95(human_rows, ai_rows, st.TELL_IDS),
        "any8": any_tell_over_p95(human_rows, ai_rows, FORWARD_TELLS),
        "any3": any_tell_over_p95(human_rows, ai_rows, HEADLINE_TELLS),
    }
    return auc, any_rule


# ---------------------------------------------------------------------------

def render(auc: Dict[str, float],
           any_rule: Dict[str, Tuple[float, float]]) -> str:
    width = max(len(label) for _, label, _ in PUBLISHED_AUC)
    lines = [
        "Combined-score aggregation sensitivity (AUC = P(AI > human window)).",
        "*oracle* = the subset can only be chosen by someone who already knows",
        "which documents are AI; an accuser with the circulating list cannot.",
        "Unmarked subsets could be chosen in advance, without the labels.",
        "",
        "  {0}  {1:>7}".format("variant".ljust(width), "AUC"),
        "  {0}  {1}".format("-" * width, "-" * 7),
    ]
    for key, label, _ in PUBLISHED_AUC:
        lines.append("  {0}  {1:>7.4f}".format(label.ljust(width), auc[key]))
    lines += [
        "",
        "Accuser's rule: flag a document if ANY tell exceeds the human 95th",
        "percentile (the shape of a real accusation — one tell is enough).",
        "",
        "  {0}  {1:>10}  {2:>13}".format("tells used".ljust(24),
                                         "AI flagged", "novelist flagged"),
        "  {0}  {1}  {2}".format("-" * 24, "-" * 10, "-" * 13),
    ]
    for key, label, _, _ in PUBLISHED_ANY_RULE:
        ai_share, human_share = any_rule[key]
        lines.append("  {0}  {1:>9.1f}%  {2:>12.1f}%".format(
            label.ljust(24), ai_share * 100, human_share * 100))
    return "\n".join(lines)


def reproduce(auc: Dict[str, float],
              any_rule: Dict[str, Tuple[float, float]]) -> int:
    print("Verifying the published sensitivity table against a recomputation")
    print("from the bundled rows. Tolerance 1e-9.")
    print()
    n_pass = n_fail = 0

    def check(label: str, got: float, want: float) -> None:
        nonlocal n_pass, n_fail
        if abs(got - want) <= 1e-9:
            n_pass += 1
            print("PASS  {0}".format(label))
        else:
            n_fail += 1
            print("FAIL  {0}\n      recomputed={1!r}\n      published= {2!r}"
                  .format(label, got, want))

    for key, label, want in PUBLISHED_AUC:
        check("auc[{0}]  {1}".format(key, label.replace("  *oracle*", "")),
              auc[key], want)
    for key, label, want_ai, want_human in PUBLISHED_ANY_RULE:
        check("any_tell>p95[{0}]: AI flagged".format(key),
              any_rule[key][0], want_ai)
        check("any_tell>p95[{0}]: novelist flagged".format(key),
              any_rule[key][1], want_human)

    print()
    if n_fail:
        print("RESULT: {0} PASS, {1} FAIL".format(n_pass, n_fail))
        return 1
    print("RESULT: all {0} sensitivity variants PASS".format(n_pass))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recompute the folk-tells combined score under every "
                    "subset and pooling variant, from the bundled rows.")
    parser.add_argument("--reproduce", action="store_true",
                        help="verify the recomputation against the published "
                             "sensitivity table; nonzero exit on mismatch")
    parser.add_argument("--data", type=Path, default=DATA_FILE,
                        help="path to folk_tells_results.json "
                             "(default: alongside this script)")
    args = parser.parse_args(argv)

    if not args.data.is_file():
        print("error: data file not found: {0}".format(args.data),
              file=sys.stderr)
        return 2
    try:
        data = json.loads(args.data.read_text(encoding="utf-8"))
    except OSError as e:
        print("error: cannot read data file {0}: {1}".format(
            args.data, e.strerror or e), file=sys.stderr)
        return 2
    except json.JSONDecodeError as e:
        print("error: malformed JSON in {0}: {1}".format(args.data, e),
              file=sys.stderr)
        return 2

    auc, any_rule = compute(data)
    if args.reproduce:
        return reproduce(auc, any_rule)
    print(render(auc, any_rule))
    return 0


if __name__ == "__main__":
    sys.exit(main())
