# Folk "AI Tells" — Scorer and Published Aggregates

This is the operationalized "AI tells" study from the preprint
(*The Width of a Voice*, v0.5.5, finding 4), packaged so you can check
it yourself.
Twelve popular tells — the em dash, "not X, but Y," rule-of-three triads,
"delve"/"leverage," corporate jargon, hedging, and the rest of the
circulating checklists — were counted with simple fixed regexes in 390
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
behaves as a model-*specific* variable, not an AI constant. The spread is
not between families but inside them — the bundled `per_model_medians`
run from 2.57 to 7.01 per 1,000 words across models of one family, and
one model uses no em dashes at all (median 0.00). Knowing which model
wrote a text would help; knowing that a machine did tells you little.

## How much does the aggregation choice matter?

A lot, and in both directions — so the whole sweep is published here
rather than left for someone else to find. The headline combines all
twelve tells in the folk direction with z-scores anchored on the human
windows. Other poolings and other subsets of the same twelve counters,
on the same rows:

| Aggregation | AUC |
|---|---|
| all 12, human-anchored z-sum — **published** | 0.5058 |
| all 12, pooled-anchored z-sum | 0.4510 |
| all 12, rank-sum pooling | 0.4417 |
| all 12, log1p rates, z-sum | 0.4572 |
| 8 forward tells (drop the 4 that ran human-high) *oracle* | 0.6657 |
| 8 forward tells, rank-sum pooling *oracle* | 0.6811 |
| 8 forward tells, log1p rates, z-sum *oracle* | 0.6549 |
| 3 headline tells: em dash / not-X-but-Y / tricolon *oracle* | 0.7056 |
| named triple: em dash / not-X-but-Y / delve | 0.7275 |
| best pair: em dash + not-X-but-Y *oracle* | 0.7343 |
| em dash alone | 0.6804 |

```
python3 sensitivity_variants.py              # the table above
python3 sensitivity_variants.py --reproduce  # verify it, exit nonzero on drift
```

Reading it honestly:

- **Why the published number is the all-twelve sum.** The folk claim is
  a *list* claim — these habits mark AI text — and the list is what gets
  applied to an accused writer. Scoring the list as a list is the test of
  the claim that was made. Every alternative pooling of all twelve
  (pooled-anchored, rank-sum, log1p) lands at or below chance, 0.44–0.51.
- ***oracle* means the subset needs the answer key.** "Drop the tells
  that run backwards" and "keep the best pair" are choices you can only
  make after seeing which documents are machine-written. A real accuser
  has the circulating checklist, not the labels. Reported here because a
  skeptic should be able to see the ceiling, not because it is a rule
  anyone could have applied in advance.
- **Two rows are not oracles.** The named triple (0.7275) is the set
  this project's own repository description used to name, and the em
  dash alone (0.6804) is the most famous tell there is; either could be
  picked in advance, and both beat chance here. That is the honest
  counter-argument to the headline, so it is printed in the same table.
  What it supports is "the circulating list is a coin flip and its best
  member is a weak, model-specific signal," not "these features are
  noise."
- **The strong tells were reported individually all along** — they are
  the first rows of the per-tell table in *The numbers*, cluster CIs
  included, and the em dash's CI [0.483, 0.849] crosses chance. At the
  threshold that catches half the machine samples it also flags 28.7% of
  the novelist windows, and its per-model medians run 0.00–7.01.
- **0.73 is a real signal and still not an accusation.** A subset that
  separates two corpora at 0.73 does not license a verdict on one
  document, which is the thing an accused writer is facing.

That last point is worth making concrete. Under the rule an accuser
actually applies — flag the document if *any* tell exceeds the novelist
95th percentile — the numbers are:

| Tells used | AI flagged | Novelist windows flagged |
|---|---|---|
| all 12 | 52.8% | 34.9% |
| 8 forward | 51.7% | 23.8% |
| 3 headline | 44.2% | 12.8% |

One in eight celebrated-novelist windows fails even the narrowest and
most favorable version of the test — and that version is the one no
accuser can construct without already knowing the answer.

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
- **Counters are simple, and simple cuts both ways.** Bounded-span
  regexes, blocklists, and framing requirements make these counters miss
  things — that part was intended. They also let some things through
  that the tell was not meant to catch, which was not: an audit of the
  counters against the corpora found real overcounting in three of the
  twelve. Each one is quantified in *What the counters actually count*
  below, with its measured effect on the published AUCs. The exclusions
  are documented inline in `score_tells.py` and pinned by
  `test_score_tells.py`.
- **This is not a detector, in either direction.** A low score does not
  clear a text and a high score does not convict one. The study's point
  is that the checklist cannot do either job.

## What the counters actually count

An adversarial audit of the counters (2026-08-06) measured where each
one departs from the tell it is named for. Nothing below changes the
shipped counters or the published numbers — the numbers are what they
are, and a counter changed after the fact would make them
unreproducible. What follows is the measured size and direction of each
defect, so a reader can discount the results correctly rather than
discover the defects later.

These measurements run over the study corpora, not over the bundled
rates, so they are reported here rather than reproducible from this
repo: the human side is copyrighted novels that are not redistributed.
Each is stated precisely enough to re-run against any corpus you do
have, and the counters they audit are the ones in `score_tells.py`.
(The pooling and subset variants in the previous section are a different
matter — those *are* reproducible here, from the bundled rows.)

**`not_x_but_y` overcounts ordinary negation.** The pattern is any
"not … , but" within a 40-character span, which catches the contrastive
reframing the tell is about ("It was not anger, but grief") *and* plain
finite-clause negation ("He did not want to go, but he went" — counted
as 1). Measured: 459 of 1,474 matches across the 78 novels (31.1%) and
75 of 450 across the 400 AI samples (16.7%) are ordinary negation rather
than reframing. The overcount is about twice as heavy on the novelists,
which means it runs *toward* this study's own negative conclusion.
Restricting the counter to
reframing-only (dropping matches whose "but"-clause has a finite subject
and verb) moves the tell from AUC 0.6211 to 0.6182, and the combined
all-twelve score from 0.5058 to 0.5116 — the coin flip survives the
correction, marginally weakened.

**`tricolon` is a serial-comma-triad proxy, not a rule-of-three
detector.** It requires the serial (Oxford) comma and 1–2 word items,
so "red, white and blue" is missed, as is any triad with longer items or
clause-length members. Read the row as "serial-comma triads with short
items"; it is as much a punctuation-style measure as a rhetorical one.

**The `-est` blocklist leaks non-superlatives.** The blocklist is a
fixed word list, so `-est` tokens outside it are counted as superlatives
whether or not they are: *unrest*, *Budapest*, *Bucharest*,
*palimpsest*, and — from a pass over every unblocked `-est` token in
both corpora — proper nouns and compounds like *Ernest*, *Forrest*,
*Suncrest*, *Pinterest*, *armrest*, *headrest*, *houseguest*. The four
named tokens are 8 of 1,506 human superlative counts (0.53%) and 9 of
759 AI counts (1.19%); hand-classifying every unblocked `-est` type puts
the total false-superlative share at 41/1,506 (2.7%) human and 29/759
(3.8%) AI. The effect on the result is immaterial: dropping the four
named tokens moves the tell from AUC 0.3049 to 0.3035, and dropping
every hand-classified non-superlative moves it to 0.3067. It stays one
of the strongly backwards tells.

**`exclamation` counts marks inside quoted dialogue.** The counter sees
punctuation, not who is speaking, and the two corpora differ in how much
dialogue they contain: 14.5% of characters in the novelist windows fall
inside quotation marks versus 9.2% in the AI samples, and 57.2% of the
novelists' exclamation marks are inside dialogue versus 91.4% of the
machines'. So the strongest backwards tell is partly measuring register.
Counting narration only (a paired-quote heuristic, exclamation marks
inside quotes ignored) moves it from AUC 0.2376 to 0.3263: still
backwards, still nowhere near a usable AI signal, but roughly a third of
the distance to chance was dialogue rather than authorship.

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

30 tests: the counter behaviors and documented exclusions, the
AUC/threshold/percentile machinery, checks that both `--reproduce` paths
exit 0 against the bundled JSON and fail on tampered rows, the pooling
variants, and the text-scoring path.

## License

Code (`score_tells.py`, `sensitivity_variants.py`, `test_score_tells.py`):
MIT, per the repository [LICENSE](../LICENSE).

Data (`folk_tells_results.json`): dedicated to the public domain under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) — see
[LICENSE-DATA](LICENSE-DATA) and the `license` field in the file's meta
block. Use it for anything; attribution appreciated but not required.
