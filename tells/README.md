# Folk "AI Tells" — Scorer and Published Aggregates

This is the operationalized "AI tells" study from the forthcoming paper
(*The Width of a Voice*, finding 4), packaged so you can check it yourself.
Twelve popular tells — the em dash, "not X, but Y," rule-of-three triads,
"delve"/"leverage," corporate jargon, hedging, and the rest of the
circulating checklists — were counted with conservative regexes in 390
windows of celebrated novelists' prose (15 authors, 78 works, 3,500 words
per window) and 400 unprompted AI long-form fiction samples (8 models,
truncated to 3,500 words). "Unprompted" means no author or style
instruction of any kind — each model was given only a neutral scenario
seed to write from (the bundled rows carry the scenario field).

The headline: **at document level, the tells cannot tell novelists from
machines.** The combined score over all 12 tells is a coin flip
(AUC 0.506), and a threshold tuned to catch half the machine samples
falsely flags 50.8% of the novelist windows.

## The numbers

Rates are per 1,000 words; medians over windows/samples. AUC = P(AI >
human window), ties 0.5, scored in the folk direction (AI-high) for every
tell — so AUC < 0.5 means the tell runs *human*-high. CI = seeded
author/model cluster bootstrap (2,000 resamples). AI sample lengths vary
(1,510–3,500 words after truncation) while human windows are a fixed
3,500 words; rates are per 1,000 words, so the two sides are comparable.

| Tell | Human med | AI med | AUC [cluster CI] | Sens @95% spec | Human flagged @50% AI |
|---|---|---|---|---|---|
| em_dash | 1.71 | 3.71 | 0.680 [0.483, 0.849] | 20.8% | 28.7% |
| not_x_but_y | 0.00 | 0.29 | 0.621 [0.521, 0.733] | 23.0% | 42.8% |
| tricolon | 0.43 | 0.57 | 0.542 [0.416, 0.664] | 5.5% | 50.0% |
| exclamation | 0.29 | 0.00 | 0.238 [0.160, 0.322] | 0.2% | degenerate† |
| lets_opener | 0.00 | 0.00 | 0.446 [0.418, 0.472] | 1.8% | degenerate† |
| superlative | 0.86 | 0.57 | 0.305 [0.225, 0.386] | 0.5% | 80.0% |
| delve_leverage | 0.00 | 0.00 | 0.496 [0.491, 0.501] | 0.2% | degenerate† |
| corporate_jargon | 0.00 | 0.00 | 0.496 [0.478, 0.512] | 3.2% | degenerate† |
| hedges | 0.57 | 0.29 | 0.410 [0.297, 0.527] | 1.8% | 74.6% |
| staging_adverbs | 0.29 | 0.29 | 0.587 [0.523, 0.649] | 6.2% | 50.3% |
| container_words | 0.00 | 0.00 | 0.496 [0.481, 0.511] | 3.0% | degenerate† |
| unnamed_consensus | 0.00 | 0.00 | 0.484 [0.454, 0.517] | 4.0% | degenerate† |
| **combined z-sum** | −0.50 | −0.52 | 0.506 [0.377, 0.623] | 5.0% | 50.8% |

† degenerate: the tell is absent from at least half the AI samples (AI
median 0), so the only threshold catching 50% of the AI flags every
document, human or machine. Six of the twelve tells are too rare in
unprompted machine fiction to flag anything.

Four tells run materially *backwards* — the novelists out-score the
machines on exclamation marks, superlatives, hedges, and "Let's" openers
(for "Let's" openers both medians are 0.00; the backwards call rests on
the AUC, 0.446, whose cluster CI excludes 0.5).
The best single tell, the em dash, has a cluster CI that crosses 0.5: it
behaves as a model-family variable (some model families are em-dash-heavy,
others use none), not an AI constant.

## Score your own prose

Pure Python standard library, 3.9+. No install, no dependencies:

```
python3 score_tells.py your_chapter.txt
```

```
your_chapter.txt — 3,482 words (rates per 1,000 words)
tell               your rate  human-novelist median  AI median
-----------------  ---------  ---------------------  ---------
em_dash                 2.30                   1.71       3.71
not_x_but_y             0.29                   0.00       0.29
tricolon                0.57                   0.43       0.57
...

NOTE — this is not a detector.
The published finding behind these medians is that the folk tells
barely separate celebrated novelists from machine fiction at document
level: the combined score over all 12 tells has AUC 0.506, a coin
flip, and a threshold tuned to catch half the AI samples falsely
flags 51% of 3,500-word windows by celebrated novelists.
...
```

Multiple files work; with no arguments it reads stdin. Below ~1,000 words
it warns that per-1,000-word rates are unstable (the published statistics
are for ~3,500-word documents).

## Reproduce the published statistics

The bundled `folk_tells_results.json` carries the per-window/per-sample
rates the paper's numbers were computed from. Recompute and verify:

```
python3 score_tells.py --reproduce
```

This recomputes, from the bundled rows, every deterministic statistic —
per-tell medians and IQRs, AUC (rank-based Mann-Whitney, ties 0.5),
the 95%-specificity threshold and the sensitivity at it, the witch-hunt
numbers (threshold catching ≥50% of AI, share of novelist windows
flagged, most-flagged authors), the combined z-sum and all of the same
for it, and the per-model medians — and prints a PASS/FAIL line per
statistic against the stored values (tolerance 1e-9). It exits nonzero
on any mismatch. 179 statistics; all PASS.

Two things are shipped as published rather than recomputed: the
cluster-bootstrap confidence intervals (they depend on NumPy's seeded
RNG; the generating tool lives in the paper repository) and the
descriptive per-author whole-work medians (they require the source
texts, which are copyrighted novels and are not redistributed).

## Scope and caveats

- **Long-form fiction at document scale.** The human side is celebrated
  literary novelists; the AI side is unprompted long-form fiction. Claims
  here are not claims about chat text, student essays, or short passages.
- **The tells were coined about chat/essay register.** Scoring them on
  fiction is exactly the point — that is where they get deployed when
  someone accuses a novelist — but the negative result is scoped to that
  use. Chat-register tells proper ("Great question," "As an AI," "I hope
  this helps") never occurred in either corpus: zero occurrences in 390
  human windows and 400 AI samples. The scorer notes them if it sees
  them but they were never scored.
- **Counters are deliberately conservative.** Bounded-span regexes,
  blocklists, and framing requirements mean undercounting is possible
  and silent overcounting is not. The exclusions are documented inline
  in `score_tells.py` and pinned by `test_score_tells.py`.
- **This is not a detector, in either direction.** A low score does not
  clear a text and a high score does not convict one. The study's point
  is that the checklist cannot do either job.

## Data note

`folk_tells_results.json` contains aggregates only: per-tell statistics,
per-model medians, per-author whole-work medians, and the per-window rows
behind them. The rows are per-1,000-word rates plus published-work
metadata (author, title, window index) on the human side and sample
metadata (model, scenario, word count) on the AI side. **No source text
is included or recoverable** — the human works are copyrighted novels and
only counts derived from them are released.

## Tests

```
python3 -m pytest test_score_tells.py    # if you have pytest
python3 test_score_tells.py              # stdlib unittest fallback
```

26 tests: the counter behaviors and documented exclusions, the
AUC/threshold/percentile machinery, a check that `--reproduce` exits 0
against the bundled JSON, and the text-scoring path.

## License

Code (`score_tells.py`, `test_score_tells.py`): MIT, per the repository
[LICENSE](../LICENSE).

Data (`folk_tells_results.json`): dedicated to the public domain under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) — see
[LICENSE-DATA](LICENSE-DATA) and the `license` field in the file's meta
block. Use it for anything; attribution appreciated but not required.
