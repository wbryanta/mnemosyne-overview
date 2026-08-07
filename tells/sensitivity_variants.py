#!/usr/bin/env python3
"""Aggregation sensitivity for the folk-tells combined score.

The published headline — the twelve tells combined score AUC 0.506, a
coin flip — pools the tells one particular way: per-tell z-scores against
the human-window mean/sd, summed in the folk direction (AI-high) over all
twelve. That is the honest test of the folk claim, because the claim on
offer is that *all of these* tells indicate AI. But it is one of several
defensible ways to pool twelve counters, and the choice moves the number
a lot. Rather than leave that for someone else to discover, this script
recomputes the combined AUC from the same bundled rows that
`score_tells.py --reproduce` verifies: a table of named variants, plus an
EXHAUSTIVE sweep of all 4,095 nonempty subsets of the twelve tells under
each of the three pooling rules, so the true ceiling is measured rather
than guessed at from a hand-picked list.

What the sweep shows: the best subset any pooling can reach is AUC 0.741
(human-anchored z-sum; 0.729 rank-sum, 0.730 log1p — all three maximized
by the same triple, em dash + not-X-but-Y + staging adverbs). That is an
ORACLE number: the maximizing subset is identifiable only by someone who
already holds the labels, and it is reported here as the ceiling a
skeptic should be able to see, not as a rule anyone could apply. Three
variants below are marked oracle-FREE, because their subsets were nameable
in advance and they still beat chance: the em dash alone, the three tells
this project's own repository description used to name, and the three
tells this project's own frozen README highlights first. Those reach
0.71-0.73 — close to the oracle ceiling, which is the honest
counter-argument to the headline and is why they are printed in the same
table. What they support is that the list is a coin flip and its best
member is a weak, model-specific signal — the em dash's cluster CI
crosses chance, and its per-model medians run 0.00-7.01. Both are
reported per-tell in README.md.

  python3 sensitivity_variants.py
      Print the variant table, the exhaustive-sweep maxima, and the
      accuser's-rule table.

  python3 sensitivity_variants.py --reproduce
      Recompute every variant and the exhaustive maxima, verify them
      against this module's pinned values, and verify that the tables
      published in README.md and ../README.md render those same values.
      PASS/FAIL per check, nonzero exit on any mismatch (tolerance 1e-9;
      the READMEs are compared at their published precision).

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
import itertools
import json
import math
import re
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
# Label-FREE by provenance: these are the first three tells highlighted in
# this project's own frozen README (tag paper-freeze-2026-07, finding 4:
# "the em dash, 'not X, but Y,' rule-of-three triads, ..."), and in the
# opening paragraph of README.md. A reader of either document picks this
# triple without seeing a single label.
HEADLINE_TELLS = ["em_dash", "not_x_but_y", "tricolon"]
BEST_PAIR = ["em_dash", "not_x_but_y"]
# The three tells the project's own GitHub description used to name.
NAMED_TRIPLE = ["em_dash", "not_x_but_y", "delve_leverage"]

# The exhaustive sweep: (variant key, pooling rule). Each variant reports
# the best AUC reachable by ANY of the 4,095 nonempty subsets of the twelve
# tells under that pooling.
EXHAUSTIVE_VARIANTS: Tuple[Tuple[str, str], ...] = (
    ("oracle_max_zsum", "zsum_human"),
    ("oracle_max_ranksum", "ranksum"),
    ("oracle_max_log1p", "log1p_zsum"),
)

# The maximizing subset, pinned. All three poolings are maximized by the
# same triple; --reproduce fails if a recomputation lands anywhere else.
PUBLISHED_ORACLE_MEMBERS: Dict[str, Tuple[str, ...]] = {
    "oracle_max_zsum": ("em_dash", "not_x_but_y", "staging_adverbs"),
    "oracle_max_ranksum": ("em_dash", "not_x_but_y", "staging_adverbs"),
    "oracle_max_log1p": ("em_dash", "not_x_but_y", "staging_adverbs"),
}


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


def _midranks(values: Sequence[float]) -> List[float]:
    """Midranks (ties averaged) of one tell's values over a pooled sample."""
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
    return rk


def rank_sum(human_rows: List[dict], ai_rows: List[dict],
             tells: Sequence[str]) -> Tuple[List[float], List[float]]:
    """Sum of within-tell midranks over the pooled sample — scale-free, so
    a single heavy-tailed tell cannot dominate the sum."""
    rows = human_rows + ai_rows
    ranks = {t: _midranks([r[t] for r in rows]) for t in tells}
    total = [math.fsum(ranks[t][i] for t in tells) for i in range(len(rows))]
    return total[:len(human_rows)], total[len(human_rows):]


def log1p_rows(rows: List[dict]) -> List[dict]:
    """log1p the rates before pooling — the usual fix for count data whose
    variance tracks its mean."""
    return [{t: math.log1p(r[t]) for t in st.TELL_IDS} for r in rows]


# ---------------------------------------------------------------------------
# Exhaustive subset sweep
#
# Every pooling above is a per-tell transform summed over the subset, and
# each tell's transform depends only on that tell's own values over the
# full sample -- never on which other tells are in the subset. So the
# per-tell contribution vectors can be computed once and the 4,095 subset
# scores read off as sums of them, which is what makes an exhaustive sweep
# cheap. The summation is math.fsum over the same doubles the named-variant
# path sums, so a subset scored here and the same subset scored through
# z_sum()/rank_sum() agree bit for bit (pinned by the test suite).
# ---------------------------------------------------------------------------

def pooling_components(
        human_rows: List[dict], ai_rows: List[dict],
        pooling: str) -> Dict[str, Tuple[List[float], List[float]]]:
    """Per-tell (human, ai) contribution vectors for one pooling rule."""
    if pooling == "ranksum":
        rows = human_rows + ai_rows
        n_human = len(human_rows)
        out = {}
        for t in st.TELL_IDS:
            rk = _midranks([r[t] for r in rows])
            out[t] = (rk[:n_human], rk[n_human:])
        return out
    if pooling == "zsum_human":
        human, ai = human_rows, ai_rows
    elif pooling == "log1p_zsum":
        human, ai = log1p_rows(human_rows), log1p_rows(ai_rows)
    else:
        raise ValueError("unknown pooling: {0!r}".format(pooling))
    out = {}
    for t in st.TELL_IDS:
        mu = st._mean([r[t] for r in human])
        sd = st._std_ddof1([r[t] for r in human]) or 1.0
        out[t] = ([(r[t] - mu) / sd for r in human],
                  [(r[t] - mu) / sd for r in ai])
    return out


def subset_auc(components: Dict[str, Tuple[List[float], List[float]]],
               tells: Sequence[str]) -> float:
    """AUC of one subset under the pooling whose components were passed."""
    parts = [components[t] for t in tells]
    human = [math.fsum(p[0][i] for p in parts)
             for i in range(len(parts[0][0]))]
    ai = [math.fsum(p[1][i] for p in parts) for i in range(len(parts[0][1]))]
    return st.mann_whitney_auc(ai, human)


def exhaustive_max(human_rows: List[dict], ai_rows: List[dict],
                   pooling: str) -> Tuple[float, Tuple[str, ...]]:
    """Best AUC over ALL 4,095 nonempty subsets of the twelve tells.

    This is the oracle ceiling: the returned subset is the one a scorer
    who already knows which documents are machine-written would choose,
    which is exactly why no accuser can choose it. Ties are broken toward
    the smaller subset, then toward the order of score_tells.TELL_IDS, so
    the answer is deterministic.
    """
    components = pooling_components(human_rows, ai_rows, pooling)
    best_auc, best_tells = -1.0, ()
    for size in range(1, len(st.TELL_IDS) + 1):
        for combo in itertools.combinations(st.TELL_IDS, size):
            auc = subset_auc(components, combo)
            if auc > best_auc:
                best_auc, best_tells = auc, combo
    return best_auc, best_tells


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
    ("headline3", "3 headline tells: em dash/not-X-but-Y/tricolon",
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
    ("oracle_max_zsum",
     "exhaustive maximum over all 4,095 subsets, z-sum   *oracle*",
     0.7408750000000000),
    ("oracle_max_ranksum",
     "exhaustive maximum over all 4,095 subsets, rank-sum   *oracle*",
     0.7285448717948718),
    ("oracle_max_log1p",
     "exhaustive maximum over all 4,095 subsets, log1p z-sum   *oracle*",
     0.7300352564102565),
]

# (key, label, published AI flagged share, published human flagged share)
PUBLISHED_ANY_RULE: List[Tuple[str, str, float, float]] = [
    ("any12", "all 12 tells", 0.5275, 0.34871794871794873),
    ("any8", "8 forward tells", 0.5175, 0.23846153846153847),
    ("any3", "3 headline tells", 0.4425, 0.1282051282051282),
]


def compute(data: dict) -> Tuple[Dict[str, float],
                                 Dict[str, Tuple[float, float]],
                                 Dict[str, Tuple[str, ...]]]:
    """Every variant, from the bundled per-window/per-sample rate rows.

    Returns (auc by variant key, any-rule shares by key, maximizing subset
    by exhaustive-variant key).
    """
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
    oracle_members: Dict[str, Tuple[str, ...]] = {}
    for key, pooling in EXHAUSTIVE_VARIANTS:
        auc[key], oracle_members[key] = exhaustive_max(
            human_rows, ai_rows, pooling)
    any_rule = {
        "any12": any_tell_over_p95(human_rows, ai_rows, st.TELL_IDS),
        "any8": any_tell_over_p95(human_rows, ai_rows, FORWARD_TELLS),
        "any3": any_tell_over_p95(human_rows, ai_rows, HEADLINE_TELLS),
    }
    return auc, any_rule, oracle_members


# ---------------------------------------------------------------------------

def render(auc: Dict[str, float],
           any_rule: Dict[str, Tuple[float, float]],
           oracle_members: Dict[str, Tuple[str, ...]]) -> str:
    width = max(len(label) for _, label, _ in PUBLISHED_AUC)
    lines = [
        "Combined-score aggregation sensitivity (AUC = P(AI > human window)).",
        "*oracle* = the subset can only be chosen by someone who already knows",
        "which documents are AI; an accuser with the circulating list cannot.",
        "Unmarked subsets could be chosen in advance, without the labels.",
        "The last three rows are the exhaustive ceiling: the best of ALL 4,095",
        "nonempty subsets of the twelve tells, per pooling.",
        "",
        "  {0}  {1:>7}".format("variant".ljust(width), "AUC"),
        "  {0}  {1}".format("-" * width, "-" * 7),
    ]
    for key, label, _ in PUBLISHED_AUC:
        lines.append("  {0}  {1:>7.4f}".format(label.ljust(width), auc[key]))
    lines += [
        "",
        "Maximizing subset per pooling (the subset the oracle rows use):",
    ]
    for key, _pooling in EXHAUSTIVE_VARIANTS:
        lines.append("  {0}  {1}".format(
            key.ljust(20), " + ".join(oracle_members[key])))
    lines += [
        "",
        "Provenance of the three unmarked multi-tell rows: 'em dash alone' is",
        "the most famous tell there is; 'named triple' is the set this",
        "project's own repository description used to name; '3 headline tells'",
        "is the first three tells highlighted in this project's own frozen",
        "README. None of the three requires a label to construct.",
    ]
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


# ---------------------------------------------------------------------------
# README consistency
#
# The constants above are the single source for these numbers; the tables
# in README.md and ../README.md are RENDERINGS of them. Nothing enforced
# that, so a README could drift from the code it documents and every test
# would still pass. This stage parses the published tables and requires
# them to render the constants exactly, at the precision they publish.
#
# Each row is pinned to a variant by a phrase distinctive enough to survive
# ordinary prose edits, and the pinning must be a bijection onto the rows
# actually present: a wrong number, a dropped row, an added row, and a row
# whose identifying phrase is gone all fail.
# ---------------------------------------------------------------------------

TELLS_README = HERE / "README.md"
ROOT_README = HERE.parent / "README.md"

README_AUC_ROWS: List[Tuple[Path, Dict[str, str]]] = [
    (TELLS_README, {
        "published": r"all 12, human-anchored z-sum",
        "pooled12": r"all 12, pooled-anchored z-sum",
        "ranksum12": r"all 12, rank-sum pooling",
        "log1p12": r"all 12, log1p rates",
        "forward8": r"8 forward tells \(drop",
        "ranksum8": r"8 forward tells, rank-sum",
        "log1p8": r"8 forward tells, log1p",
        "headline3": r"3 headline tells",
        "named3": r"named triple",
        "pair2": r"best pair",
        "em_dash_alone": r"em dash alone",
        "oracle_max_zsum":
            r"exhaustive maximum over all 4,095 subsets, z-sum",
        "oracle_max_ranksum":
            r"exhaustive maximum over all 4,095 subsets, rank-sum",
        "oracle_max_log1p":
            r"exhaustive maximum over all 4,095 subsets, log1p",
    }),
    (ROOT_README, {
        "published": r"human-anchored z-sum",
        "pooled12": r"pooled-anchored z-sum",
        "ranksum12": r"all twelve, rank-sum",
        "forward8": r"forward tells \(drop",
        "headline3": r"three headline tells",
        "named3": r"named triple",
        "pair2": r"best pair",
        "em_dash_alone": r"em dash alone",
        "oracle_max_zsum":
            r"exhaustive maximum over all 4,095 subsets \(z-sum\)",
        "oracle_max_ranksum":
            r"exhaustive maximum over all 4,095 subsets \(rank-sum\)",
        "oracle_max_log1p":
            r"exhaustive maximum over all 4,095 subsets \(log1p",
    }),
]

README_ANY_RULE_ROWS: Dict[str, str] = {
    "any12": r"all 12",
    "any8": r"8 forward",
    "any3": r"3 headline",
}


def _table_cells(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _markdown_tables(text: str) -> List[Tuple[List[str], List[List[str]]]]:
    """(header cells, body rows) for every pipe table in a markdown file."""
    lines = text.splitlines()
    tables = []
    i = 0
    while i < len(lines):
        is_header = (
            lines[i].lstrip().startswith("|")
            and i + 1 < len(lines)
            and re.fullmatch(r"\s*\|[\s:|-]+\|\s*", lines[i + 1]) is not None)
        if not is_header:
            i += 1
            continue
        header = _table_cells(lines[i])
        rows = []
        i += 2
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            rows.append(_table_cells(lines[i]))
            i += 1
        tables.append((header, rows))
    return tables


def _select_table(tables, predicate, what: str, path: Path):
    matched = [t for t in tables if predicate(t[0])]
    if len(matched) != 1:
        raise ValueError(
            "expected exactly 1 {0} table in {1}, found {2}".format(
                what, path.name, len(matched)))
    return matched[0]


def _pin_rows(rows: List[List[str]], patterns: Dict[str, str],
              path: Path) -> Dict[str, List[str]]:
    """Map each table row to exactly one variant key, and back again."""
    claimed: Dict[str, List[str]] = {}
    for row in rows:
        hits = [k for k, pat in patterns.items()
                if re.search(pat, row[0], re.IGNORECASE)]
        if len(hits) != 1:
            raise ValueError(
                "{0}: row {1!r} matches {2} known variants (expected 1); the "
                "published table and this module have diverged".format(
                    path.name, row[0], len(hits)))
        if hits[0] in claimed:
            raise ValueError("{0}: two rows both claim variant {1!r}".format(
                path.name, hits[0]))
        claimed[hits[0]] = row
    missing = sorted(set(patterns) - set(claimed))
    if missing:
        raise ValueError("{0}: published table is missing rows for {1}".format(
            path.name, ", ".join(missing)))
    return claimed


def check_readmes(auc_readmes: Optional[List[Tuple[Path, Dict[str, str]]]] = None,
                  any_rule_readme: Optional[Path] = None
                  ) -> List[Tuple[bool, str]]:
    """Verify the published README tables render this module's constants.

    Returns (ok, message) per check. Paths are injectable so the test suite
    can point the checker at a tampered copy.
    """
    auc_readmes = README_AUC_ROWS if auc_readmes is None else auc_readmes
    any_rule_readme = (TELLS_README if any_rule_readme is None
                       else any_rule_readme)
    published_auc = {key: value for key, _, value in PUBLISHED_AUC}
    results: List[Tuple[bool, str]] = []

    for path, patterns in auc_readmes:
        try:
            tables = _markdown_tables(path.read_text(encoding="utf-8"))
            _header, rows = _select_table(
                tables, lambda h: h[-1].strip() == "AUC", "AUC", path)
            claimed = _pin_rows(rows, patterns, path)
        except (OSError, ValueError) as exc:
            results.append((False, "{0}: {1}".format(path.name, exc)))
            continue
        for key in sorted(patterns):
            want = "%.4f" % published_auc[key]
            got = claimed[key][-1]
            results.append((
                got == want,
                "{0} table renders auc[{1}] as {2} (module: {3})".format(
                    path.name, key, got, want)))

    path = any_rule_readme
    try:
        tables = _markdown_tables(path.read_text(encoding="utf-8"))
        _header, rows = _select_table(
            tables, lambda h: any(c.startswith("AI flagged") for c in h),
            "accuser's-rule", path)
        claimed = _pin_rows(rows, README_ANY_RULE_ROWS, path)
    except (OSError, ValueError) as exc:
        results.append((False, "{0}: {1}".format(path.name, exc)))
        return results
    for key, _label, want_ai, want_human in PUBLISHED_ANY_RULE:
        row = claimed[key]
        for column, want in ((1, want_ai), (2, want_human)):
            want_str = "%.1f%%" % (want * 100)
            got = row[column]
            results.append((
                got == want_str,
                "{0} accuser's-rule table renders {1}[{2}] as {3} "
                "(module: {4})".format(path.name, key,
                                       "ai" if column == 1 else "human",
                                       got, want_str)))
    return results


def reproduce(auc: Dict[str, float],
              any_rule: Dict[str, Tuple[float, float]],
              oracle_members: Dict[str, Tuple[str, ...]]) -> int:
    print("Verifying the published sensitivity table against a recomputation")
    print("from the bundled rows. Tolerance 1e-9.")
    print()
    n_pass = n_fail = 0

    def record(ok: bool, label: str, detail: str = "") -> None:
        nonlocal n_pass, n_fail
        if ok:
            n_pass += 1
            print("PASS  {0}".format(label))
        else:
            n_fail += 1
            print("FAIL  {0}{1}".format(label, detail))

    def check(label: str, got: float, want: float) -> None:
        record(abs(got - want) <= 1e-9, label,
               "\n      recomputed={0!r}\n      published= {1!r}".format(
                   got, want))

    for key, label, want in PUBLISHED_AUC:
        check("auc[{0}]  {1}".format(key, label.replace("  *oracle*", "")),
              auc[key], want)
    for key, _pooling in EXHAUSTIVE_VARIANTS:
        want_members = PUBLISHED_ORACLE_MEMBERS[key]
        got_members = oracle_members[key]
        record(got_members == want_members,
               "maximizing subset[{0}]  {1}".format(
                   key, " + ".join(want_members)),
               "\n      recomputed={0!r}\n      published= {1!r}".format(
                   got_members, want_members))
    for key, label, want_ai, want_human in PUBLISHED_ANY_RULE:
        check("any_tell>p95[{0}]: AI flagged".format(key),
              any_rule[key][0], want_ai)
        check("any_tell>p95[{0}]: novelist flagged".format(key),
              any_rule[key][1], want_human)

    print()
    print("Verifying the published README tables render the same constants.")
    print()
    for ok, message in check_readmes():
        record(ok, message)

    print()
    if n_fail:
        print("RESULT: {0} PASS, {1} FAIL".format(n_pass, n_fail))
        return 1
    print("RESULT: all {0} sensitivity checks PASS".format(n_pass))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recompute the folk-tells combined score under the named "
                    "subset/pooling variants and under an exhaustive sweep of "
                    "all 4,095 subsets, from the bundled rows.")
    parser.add_argument("--reproduce", action="store_true",
                        help="verify the recomputation against this module's "
                             "pinned values and against the tables published "
                             "in both READMEs; nonzero exit on mismatch")
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

    auc, any_rule, oracle_members = compute(data)
    if args.reproduce:
        return reproduce(auc, any_rule, oracle_members)
    print(render(auc, any_rule, oracle_members))
    return 0


if __name__ == "__main__":
    sys.exit(main())
