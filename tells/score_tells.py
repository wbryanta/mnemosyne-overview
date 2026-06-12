#!/usr/bin/env python3
"""Folk "AI tells" scorer — score your own prose, or reproduce the paper's numbers.

Popular guidance (14-item tell lists circulating in 2024-2026) claims AI
text is identifiable from surface habits: em dashes, "not X, it's Y"
constructions, rule-of-three tricolons, "delve"/"leverage", corporate
jargon, hedging, sentence-initial "Let's", exclamation and superlative
density, staging adverbs ("quietly"), container words ("space",
"opportunity"), and unnamed-consensus appeals ("most people"). The study
this ships with operationalized twelve of those tells conservatively and
scored celebrated novelists against unprompted AI long-form fiction. The
headline: at document level the tells barely separate the two (combined
AUC 0.506), and several run in the wrong direction.

Two modes:

  python3 score_tells.py FILE [FILE ...]
      Score each text on the 12 tells as per-1,000-word rates and print
      them next to the human-novelist and AI medians from the bundled
      aggregates (folk_tells_results.json). Reads stdin if no files given.

  python3 score_tells.py --reproduce
      Recompute the published per-tell statistics from the bundled
      per-window/per-sample rate rows and verify them against the stored
      aggregates. Prints PASS/FAIL per statistic; exits nonzero on any
      mismatch. The cluster-bootstrap confidence intervals are
      RNG-dependent (NumPy Generator in the original tool) and are
      reported as published rather than recomputed.

Pure Python standard library (3.9+). No third-party packages: the point
is zero-install verification. The tell counters and the deterministic
statistics mirror the canonical analysis tool (analyze_folk_tells.py in
the paper repository) exactly; NumPy's rank/percentile conventions are
reimplemented below (stable mergesort midranks; linear-interpolation
percentiles).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

DATA_FILE = Path(__file__).resolve().parent / "folk_tells_results.json"

# ---------------------------------------------------------------------------
# Tell operationalizations. Each is a conservative regex/heuristic; rates are
# per 1,000 whitespace words. Conservatism notes inline — undercounting is
# acceptable, silent overcounting is not. All counters run on raw text
# (punctuation intact). Copied verbatim from the canonical analysis tool.
# ---------------------------------------------------------------------------

# -est words that are not superlatives (blocklist for the -est heuristic).
_EST_BLOCKLIST = {
    "honest", "modest", "earnest", "forest", "harvest", "arrest", "request",
    "suggest", "digest", "manifest", "protest", "interest", "tempest",
    "conquest", "contest", "invest", "infest", "divest", "detest", "behest",
    "bequest", "inquest", "everest", "northwest", "southwest", "midwest",
    "rest", "west", "test", "best",  # "best" handled separately below
    "guest", "chest", "quest", "vest", "jest", "nest", "pest", "zest",
    "crest", "wrest", "molest", "attest", "contests", "priest", "incest",
    "obsessed",  # safety: never matches but documents intent
    "dearest",  # term of address, not comparison -- still a superlative form;
                # kept OUT of the blocklist would be defensible; we block it
                # as vocative usage dominates in fiction
}
_EST_RE = re.compile(r"\b[A-Za-z]{3,}est\b")
_MOST_ADJ_RE = re.compile(
    r"\b(?:most|least)\s+[a-z]+"
    r"(?:ful|ous|ive|able|ible|ant|ent|al|ic|less|ish|some|ing|ed|y)\b",
    re.IGNORECASE,
)


def _count_superlatives(text: str) -> int:
    """-est superlatives (blocklist-filtered) + most/least + adjective-suffix
    heuristic + irregular 'best'/'worst'. Conservative: multiword adjectives
    and bare 'most + noun' are not counted ('most people' is its own tell)."""
    n = 0
    for m in _EST_RE.finditer(text):
        w = m.group(0).lower()
        if w in _EST_BLOCKLIST:
            continue
        n += 1
    n += len(re.findall(r"\b(?:best|worst)\b", text, re.IGNORECASE))
    n += len(_MOST_ADJ_RE.findall(text))
    return n


# "not X, but Y" / "not X — it's Y" contrastive reframing. Two patterns,
# both bounded at 40 chars with no sentence-internal stop punctuation, so
# cross-sentence and long-range matches are excluded. "not only ... but
# also" (classical rhetoric) is excluded explicitly.
_NOT_BUT_RE = re.compile(
    r"\bnot\b(?!\s+only\b)[^.?!;:—]{0,40}?,\s*but\b", re.IGNORECASE)
_NOT_ITS_RE = re.compile(
    r"\bnot\b[^.?!;:]{0,40}?[—,;]\s*it[’']s\b", re.IGNORECASE)


def _count_not_x_but_y(text: str) -> int:
    return len(_NOT_BUT_RE.findall(text)) + len(_NOT_ITS_RE.findall(text))


# Serial triad "X, Y, and Z" with 1-2 word items (tricolon proxy).
# Conservative: triads with longer items or no serial comma are missed.
_TRICOLON_RE = re.compile(
    r"\b[\w’']+(?:\s+[\w’']+)?,\s+[\w’']+(?:\s+[\w’']+)?,"
    r"\s+(?:and|or)\s+[\w’']+", re.IGNORECASE)

# Sentence-initial "Let's"/"Let us" (start of text, after terminal
# punctuation + optional closing quote, or on a new line).
_LETS_RE = re.compile(
    r"(?:^|[.?!][\"”’']?\s+|\n\s*)Let(?:[’']s|\s+us)\b")

_DELVE_LEVERAGE_RE = re.compile(
    r"\b(?:delv(?:e|es|ed|ing)|leverag(?:e|es|ed|ing))\b", re.IGNORECASE)

# Fixed corporate-jargon lexicon (word-bounded; multiword phrases included).
# 'leverage' is excluded here (it is its own tell); literal-use-risk words
# ('unlock', 'journey', 'robust') are excluded as too common in fiction.
_JARGON_RE = re.compile(
    r"\b(?:synerg(?:y|ies|istic)|stakeholders?|scalab(?:le|ility)"
    r"|actionable|streamlin(?:e|es|ed|ing)|seamless(?:ly)?"
    r"|holistic(?:ally)?|paradigms?|ecosystems?"
    r"|empower(?:s|ed|ing|ment)?|optimi[sz](?:e|es|ed|ing|ation)"
    r"|utili[sz](?:e|es|ed|ing)|bandwidth|deliverables?"
    r"|best\s+practices|value[-\s]add(?:ed)?|touch\s+base|circle\s+back"
    r"|deep\s+dive|game[-\s]chang(?:er|ing)|cutting[-\s]edge"
    r"|state[-\s]of[-\s]the[-\s]art|low[-\s]hanging\s+fruit"
    r"|move\s+the\s+needle|win[-\s]win|core\s+competenc(?:y|ies)"
    r"|pain\s+points?)\b", re.IGNORECASE)

# Fixed hedge lexicon. Bare modals (may/might/could) are excluded as
# uncountably polysemous; this is the essayistic-hedging subset of the
# folk lists that can be counted without parsing.
_HEDGE_RE = re.compile(
    r"\b(?:perhaps|possibly|arguably|presumably|seemingly|apparently"
    r"|somewhat|may\s+well|might\s+well|tends?\s+to"
    r"|in\s+many\s+ways|to\s+some\s+extent|more\s+often\s+than\s+not"
    r"|it\s+seems\s+that|it\s+would\s+seem)\b", re.IGNORECASE)

# Staging adverbs ("quietly devastating" register). Counted bare — the
# folk claim is about density of these adverbs, not their syntax.
_STAGING_RE = re.compile(
    r"\b(?:quietly|softly|gently|subtly|deliberately|effortlessly"
    r"|undeniably|unmistakably|profoundly)\b", re.IGNORECASE)

# Container words: abstract "space"/"opportunity" frames only — literal
# rooms and gaps are excluded by requiring the framing construction.
_CONTAINER_RE = re.compile(
    r"\b(?:h[oe]ld(?:ing)?\s+space|safe\s+spaces?"
    r"|a\s+space\s+(?:for|where|to|of|in\s+which)|the\s+space\s+to"
    r"|in\s+(?:this|that)\s+space"
    r"|(?:an?|the)\s+opportunit(?:y|ies)\s+(?:to|for)"
    r"|opportunities\s+to)\b", re.IGNORECASE)

# Unnamed-consensus appeals.
_CONSENSUS_RE = re.compile(
    r"\b(?:most\s+people|many\s+people|some\s+(?:would\s+say|say|argue)"
    r"|many\s+(?:believe|argue|say)|everyone\s+knows|people\s+often"
    r"|experts\s+(?:say|agree)|studies\s+show|we\s+all\s+know"
    r"|it\s+is\s+widely|it[’']s\s+widely)\b", re.IGNORECASE)

# Em dash: the "—" character plus standalone double-hyphen (both corpora
# use "—" natively; "--" appears residually in some shelf scans).
_DOUBLE_HYPHEN_RE = re.compile(r"(?<!-)--(?!-)")


def _count_em_dash(text: str) -> int:
    return text.count("—") + len(_DOUBLE_HYPHEN_RE.findall(text))


# Chat-register tells: cannot plausibly occur inside fiction; counted as an
# out-of-register note only, never scored. (In the published corpora these
# occurred zero times on both sides.)
_CHAT_REGISTER_RE = re.compile(
    r"\b(?:Great\s+question|I\s+hope\s+this\s+helps|As\s+an\s+AI"
    r"|I[’']m\s+happy\s+to\s+help|Certainly!|I\s+cannot\s+assist)\b")

# (tell_id, gloss, counter). Folk direction for every tell is AI-high.
TELLS: List[Tuple[str, str, Callable[[str], int]]] = [
    ("em_dash", "em dashes (— plus standalone --)", _count_em_dash),
    ("not_x_but_y", "contrastive reframing: 'not X, but Y' / 'not X — it's Y'",
     _count_not_x_but_y),
    ("tricolon", "serial triad 'X, Y, and Z' (1-2 word items; rule-of-three proxy)",
     lambda t: len(_TRICOLON_RE.findall(t))),
    ("exclamation", "exclamation marks", lambda t: t.count("!")),
    ("lets_opener", "sentence-initial 'Let's' / 'Let us'",
     lambda t: len(_LETS_RE.findall(t))),
    ("superlative", "superlatives (-est blocklist-filtered; best/worst; most/least+adj)",
     _count_superlatives),
    ("delve_leverage", "'delve'/'leverage' lemmas",
     lambda t: len(_DELVE_LEVERAGE_RE.findall(t))),
    ("corporate_jargon", "fixed corporate-jargon lexicon (synergy, stakeholder, ...)",
     lambda t: len(_JARGON_RE.findall(t))),
    ("hedges", "fixed hedge lexicon (perhaps, arguably, seemingly, ...)",
     lambda t: len(_HEDGE_RE.findall(t))),
    ("staging_adverbs", "staging adverbs (quietly, softly, gently, profoundly, ...)",
     lambda t: len(_STAGING_RE.findall(t))),
    ("container_words", "abstract 'space'/'opportunity' frames",
     lambda t: len(_CONTAINER_RE.findall(t))),
    ("unnamed_consensus", "unnamed-consensus appeals ('most people', 'studies show', ...)",
     lambda t: len(_CONSENSUS_RE.findall(t))),
]

TELL_IDS = [tid for tid, _, _ in TELLS]


def rates_for(text: str, n_words: int) -> Dict[str, float]:
    per_k = 1000.0 / max(n_words, 1)
    return {tid: counter(text) * per_k for tid, _, counter in TELLS}


# ---------------------------------------------------------------------------
# Statistics — stdlib reimplementations of the canonical tool's NumPy math.
# Rank and percentile conventions are matched exactly:
# - mann_whitney_auc: stable-sort ranks with midranks for ties (rank sums
#   over half-integers are exact in float64, so the AUC reproduces bit-for-
#   bit against the NumPy original).
# - percentile: NumPy's default linear interpolation, including its lerp
#   branch (b - (b-a)*(1-t) when t >= 0.5).
# - mean/std use math.fsum (exactly rounded); NumPy uses pairwise summation.
#   The two agree to ~1 ulp, far inside the 1e-9 verification tolerance.
# ---------------------------------------------------------------------------

def _median(values: Sequence[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        raise ValueError("median of empty sequence")
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2


def _percentile(values: Sequence[float], q: float) -> float:
    """NumPy default ('linear') percentile, q in [0, 100]."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        raise ValueError("percentile of empty sequence")
    virtual = (q / 100.0) * (n - 1)
    lo = math.floor(virtual)
    hi = math.ceil(virtual)
    a, b = s[int(lo)], s[int(hi)]
    t = virtual - lo
    if t >= 0.5:  # NumPy's lerp uses the b-anchored form here
        return b - (b - a) * (1 - t)
    return a + (b - a) * t


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _std_ddof1(values: Sequence[float]) -> float:
    mu = _mean(values)
    return math.sqrt(math.fsum((v - mu) ** 2 for v in values) / (len(values) - 1))


def mann_whitney_auc(ai: Sequence[float], human: Sequence[float]) -> float:
    """P(AI > human) + 0.5 P(tie), via rank sums (midranks for ties)."""
    combined = list(ai) + list(human)
    n = len(combined)
    order = sorted(range(n), key=combined.__getitem__)  # stable, like mergesort
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and combined[order[j + 1]] == combined[order[i]]:
            j += 1
        midrank = (i + j + 2) / 2  # 1-based ranks i+1 .. j+1, averaged
        for k in range(i, j + 1):
            ranks[order[k]] = midrank
        i = j + 1
    r_ai = sum(ranks[: len(ai)])
    u = r_ai - len(ai) * (len(ai) + 1) / 2
    return u / (len(ai) * len(human))


def _author_display(slug: str) -> str:
    surname = slug.split("-")[0].replace("_", " ")
    return " ".join(p.capitalize() for p in surname.split())


def tell_stats(ai: Sequence[float], human: Sequence[float],
               human_authors: Sequence[str]) -> dict:
    """Deterministic subset of the canonical tell_stats (no bootstrap CI)."""
    auc = mann_whitney_auc(ai, human)
    # 95% specificity on human windows: flag if rate > human p95 (strict >,
    # conservative for the detector). Report achieved specificity exactly.
    thr95 = _percentile(human, 95)
    spec_achieved = sum(1 for v in human if v <= thr95) / len(human)
    sens_at_95spec = sum(1 for v in ai if v > thr95) / len(ai)
    # Witch-hunt: threshold catching >=50% of AI (flag if rate >= AI median).
    # Degenerate when the AI median is 0 (tell absent from >=half the AI
    # samples): the only threshold catching 50% of AI flags every document,
    # human or machine. Flagged explicitly so the 100% is not misread.
    thr50 = _median(ai)
    # Degeneracy only applies to non-negative per-1,000-word rates; the
    # combined z-sum is signed and a negative median is a real threshold.
    degenerate = (thr50 <= 0.0 and all(v >= 0 for v in ai)
                  and all(v >= 0 for v in human))
    ai_caught = sum(1 for v in ai if v >= thr50) / len(ai)
    human_flagged_mask = [v >= thr50 for v in human]
    human_flagged = sum(human_flagged_mask) / len(human)
    by_author: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for flagged, author in zip(human_flagged_mask, human_authors):
        by_author[author][1] += 1
        if flagged:
            by_author[author][0] += 1
    flagged_authors = sorted(
        ((a, f, n, f / n) for a, (f, n) in by_author.items()),
        key=lambda t: (-t[3], t[0]))
    return {
        "human_median": _median(human),
        "human_iqr": [_percentile(human, 25), _percentile(human, 75)],
        "ai_median": _median(ai),
        "ai_iqr": [_percentile(ai, 25), _percentile(ai, 75)],
        "auc_ai_high": auc,
        "threshold_at_95pct_specificity": thr95,
        "specificity_achieved": spec_achieved,
        "sensitivity_on_ai_at_95spec": sens_at_95spec,
        "witch_hunt": {
            "threshold_flagging_50pct_ai": thr50,
            "degenerate_threshold": degenerate,
            "ai_share_caught": ai_caught,
            "human_windows_flagged_pct": human_flagged,
            "most_flagged_authors": [] if degenerate else [
                {"author": _author_display(a), "slug": a,
                 "flagged": f, "windows": n, "share": round(s, 4)}
                for a, f, n, s in flagged_authors[:5] if f > 0
            ],
        },
    }


def combined_z_rows(human_rows: List[dict], ai_rows: List[dict]
                    ) -> Tuple[List[float], List[float]]:
    """Combined z-sum per row: per-tell z against the human-window mean/sd,
    summed over all 12 tells (folk direction), exactly as the canonical tool."""
    mu = {tid: _mean([r[tid] for r in human_rows]) for tid in TELL_IDS}
    sd = {tid: _std_ddof1([r[tid] for r in human_rows]) or 1.0 for tid in TELL_IDS}
    human_z = [math.fsum((r[tid] - mu[tid]) / sd[tid] for tid in TELL_IDS)
               for r in human_rows]
    ai_z = [math.fsum((r[tid] - mu[tid]) / sd[tid] for tid in TELL_IDS)
            for r in ai_rows]
    return human_z, ai_z


# ---------------------------------------------------------------------------
# Mode A: score texts
# ---------------------------------------------------------------------------

UNSTABLE_BELOW_WORDS = 1000


def render_score_table(name: str, text: str, data: dict) -> str:
    n_words = len(re.findall(r"\S+", text))
    rates = rates_for(text, n_words)
    tells = data["tells"]
    header = ("tell", "your rate", "human-novelist median", "AI median")
    rows = [(tid, "{0:.2f}".format(rates[tid]),
             "{0:.2f}".format(tells[tid]["human_median"]),
             "{0:.2f}".format(tells[tid]["ai_median"]))
            for tid in TELL_IDS]
    widths = [max(len(header[c]), *(len(r[c]) for r in rows)) for c in range(4)]
    lines = ["{0} — {1:,} {2} (rates per 1,000 words)".format(
        name, n_words, "word" if n_words == 1 else "words")]
    lines.append("  ".join(
        h.ljust(widths[c]) if c == 0 else h.rjust(widths[c])
        for c, h in enumerate(header)))
    lines.append("  ".join("-" * w for w in widths))
    for r in rows:
        lines.append("  ".join(
            v.ljust(widths[c]) if c == 0 else v.rjust(widths[c])
            for c, v in enumerate(r)))
    chat_hits = len(_CHAT_REGISTER_RE.findall(text))
    if chat_hits:
        lines.append("note: {0} chat-register phrase(s) ('Great question', "
                     "'As an AI', ...) found — these are out of register for "
                     "prose and were never scored in the study.".format(chat_hits))
    if n_words < UNSTABLE_BELOW_WORDS:
        lines.append("WARNING: only {0:,} words; per-1,000-word rates are "
                     "unstable below ~{1:,} words. The published statistics "
                     "are for ~3,500-word documents.".format(
                         n_words, UNSTABLE_BELOW_WORDS))
    return "\n".join(lines)


def caution_block(data: dict) -> str:
    comb = data["combined_z_sum"]
    return "\n".join([
        "",
        "NOTE — this is not a detector.",
        "The published finding behind these medians is that the folk tells",
        "barely separate celebrated novelists from machine fiction at document",
        "level: the combined score over all 12 tells has AUC {0:.3f}, a coin"
        .format(comb["auc_ai_high"]),
        "flip, and a threshold tuned to catch half the AI samples falsely",
        "flags {0:.0f}% of 3,500-word windows by celebrated novelists."
        .format(comb["witch_hunt"]["human_windows_flagged_pct"] * 100),
        "Matching an 'AI median' here is not evidence of anything; several",
        "tells run higher in human prose than in machine prose.",
        "Scope: long-form prose at document scale. Short passages and",
        "chat-register text are out of scope for these numbers.",
    ])


# ---------------------------------------------------------------------------
# Mode B: --reproduce
# ---------------------------------------------------------------------------

class _Checker:
    def __init__(self) -> None:
        self.n_pass = 0
        self.n_fail = 0

    def check(self, label: str, got, want, tol: float = 1e-9) -> None:
        ok = self._eq(got, want, tol)
        if ok:
            self.n_pass += 1
            print("PASS  {0}".format(label))
        else:
            self.n_fail += 1
            print("FAIL  {0}\n      recomputed={1!r}\n      stored=    {2!r}"
                  .format(label, got, want))

    def _eq(self, got, want, tol: float) -> bool:
        if isinstance(want, float) or isinstance(got, float):
            return isinstance(got, (int, float)) and isinstance(want, (int, float)) \
                and abs(float(got) - float(want)) <= tol
        if isinstance(want, list):
            return (isinstance(got, list) and len(got) == len(want)
                    and all(self._eq(g, w, tol) for g, w in zip(got, want)))
        if isinstance(want, dict):
            return (isinstance(got, dict) and set(got) == set(want)
                    and all(self._eq(got[k], want[k], tol) for k in want))
        return got == want


def reproduce(data: dict) -> int:
    meta = data["meta"]
    human_rows = data["human_window_rows"]
    ai_rows = data["ai_sample_rows"]
    print("Reproducing published statistics from {0} human windows and {1} AI "
          "samples (bundled per-row rates)...".format(len(human_rows), len(ai_rows)))
    print("Tolerance 1e-9 on floats; counts, flags, and author lists exact.")
    print()

    ck = _Checker()
    ck.check("meta: n_human_windows", len(human_rows), meta["n_human_windows"])
    ck.check("meta: n_ai_samples", len(ai_rows), meta["n_ai_samples"])

    human_authors = [r["author"] for r in human_rows]
    targets = list(TELL_IDS) + ["combined_z_sum"]
    human_z, ai_z = combined_z_rows(human_rows, ai_rows)

    for tid in targets:
        if tid == "combined_z_sum":
            stored = data["combined_z_sum"]
            human, ai = human_z, ai_z
        else:
            stored = data["tells"][tid]
            human = [r[tid] for r in human_rows]
            ai = [r[tid] for r in ai_rows]
        got = tell_stats(ai, human, human_authors)
        ck.check("{0}: human_median".format(tid), got["human_median"],
                 stored["human_median"])
        ck.check("{0}: human_iqr".format(tid), got["human_iqr"],
                 stored["human_iqr"])
        ck.check("{0}: ai_median".format(tid), got["ai_median"],
                 stored["ai_median"])
        ck.check("{0}: ai_iqr".format(tid), got["ai_iqr"], stored["ai_iqr"])
        ck.check("{0}: auc_ai_high".format(tid), got["auc_ai_high"],
                 stored["auc_ai_high"])
        ck.check("{0}: threshold_at_95pct_specificity".format(tid),
                 got["threshold_at_95pct_specificity"],
                 stored["threshold_at_95pct_specificity"])
        ck.check("{0}: specificity_achieved".format(tid),
                 got["specificity_achieved"], stored["specificity_achieved"])
        ck.check("{0}: sensitivity_on_ai_at_95spec".format(tid),
                 got["sensitivity_on_ai_at_95spec"],
                 stored["sensitivity_on_ai_at_95spec"])
        gw, sw = got["witch_hunt"], stored["witch_hunt"]
        ck.check("{0}: witch_hunt.threshold_flagging_50pct_ai".format(tid),
                 gw["threshold_flagging_50pct_ai"],
                 sw["threshold_flagging_50pct_ai"])
        ck.check("{0}: witch_hunt.degenerate_threshold".format(tid),
                 gw["degenerate_threshold"], sw["degenerate_threshold"])
        ck.check("{0}: witch_hunt.ai_share_caught".format(tid),
                 gw["ai_share_caught"], sw["ai_share_caught"])
        ck.check("{0}: witch_hunt.human_windows_flagged_pct".format(tid),
                 gw["human_windows_flagged_pct"],
                 sw["human_windows_flagged_pct"])
        ck.check("{0}: witch_hunt.most_flagged_authors".format(tid),
                 gw["most_flagged_authors"], sw["most_flagged_authors"])
        ci = stored["auc_ci95_cluster_bootstrap"]
        print("INFO  {0}: auc_ci95_cluster_bootstrap [{1:.3f}, {2:.3f}] as "
              "published (RNG-dependent; seeded NumPy author/model cluster "
              "bootstrap — see the paper repo for the generating tool)"
              .format(tid, ci[0], ci[1]))

    # Per-model medians (recomputable from the AI sample rows).
    for model in sorted(data["per_model_medians"]):
        rows = [r for r in ai_rows if r["model"] == model]
        got_m = {tid: _median([r[tid] for r in rows]) for tid in TELL_IDS}
        ck.check("per_model_medians: {0}".format(model), got_m,
                 data["per_model_medians"][model])

    print()
    print("Note: per_author_work_medians (whole-work descriptive medians) and "
          "the bootstrap CIs require the source texts / the original NumPy "
          "RNG and are shipped as published, not recomputed here.")
    print()
    if ck.n_fail:
        print("RESULT: {0} PASS, {1} FAIL".format(ck.n_pass, ck.n_fail))
        return 1
    print("RESULT: all {0} statistics PASS".format(ck.n_pass))
    return 0


# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score prose on the 12 folk 'AI tells', or reproduce the "
                    "published statistics from the bundled aggregates.")
    parser.add_argument("files", nargs="*", type=Path,
                        help="text files to score (stdin if none)")
    parser.add_argument("--reproduce", action="store_true",
                        help="recompute the published statistics from the "
                             "bundled rows and verify against stored values")
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

    if args.reproduce:
        if args.files:
            parser.error("--reproduce takes no input files")
        return reproduce(data)

    if args.files:
        texts = []
        for f in args.files:
            try:
                texts.append(
                    (str(f), f.read_text(encoding="utf-8", errors="replace")))
            except OSError as e:
                print("error: cannot read {0}: {1}".format(
                    f, e.strerror or e), file=sys.stderr)
                return 2
    else:
        if sys.stdin.isatty():
            print("Reading from stdin (no files given); end with Ctrl-D.",
                  file=sys.stderr)
        texts = [("<stdin>",
                  sys.stdin.buffer.read().decode("utf-8", errors="replace"))]

    for i, (name, text) in enumerate(texts):
        if i:
            print()
        print(render_score_table(name, text, data))
    print(caution_block(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
