# Mnemosyne

**An applied-ML instrument for measuring authorial voice — and the research program that grew out of it: placing machine imitation inside authors' own variation.**

Mnemosyne began as a personal research project with a stubborn question: *is voice something we can only recognize, or can we actually measure it?* Most tools that touch this question are framed as AI detectors. Mnemosyne is not one, and the research it produced is explicitly anti-detector: a detector hands down a verdict; a measurement places a text in a calibrated space and shows its work, so that every claim can fail loudly.

This page is a preview. The full research release — instrument, validation harness, replication data, and a paper — is being prepared now (see *Status*, below).

## From instrument to research program

The first version of Mnemosyne fingerprinted a single writer's voice against their own corpus. Building it surfaced the deeper problem: a personal baseline is an uncalibrated origin. You cannot say how far a passage sits from *one* voice until you know how far voices sit from *each other* — and how far a single author wanders within their own work.

So the project was rebuilt around an **author-relative measurement space**: a shelf of contemporary novelists (with a public-domain replication shelf alongside it), pooled normalization, and — the load-bearing idea — **envelopes of within-author variation**, measured per author at matched text length. The unit of meaning stops being "distance from a fingerprint" and becomes *placement inside or outside an author's own measured range*. An imitation doesn't have to land on a point; it has to land inside the width of a voice. That width turns out to be the most interesting object in the system.

The measurement runs on two layers with very different behavior:

- **Texture** — lexical diversity, rhythm, repetition, affect posture: the layer a reader notices, and the layer style prompting visibly moves.
- **The chassis** — the distribution of closed-class function words ("the," "of," "but," "which"...) that classical stylometry has used for attribution since the 1960s: the layer nobody chooses consciously.

## What we found (preview)

A paper — *The Width of a Voice: Placing Machine Imitation Inside Authors' Own Variation* — is drafted and frozen against audited evidence. Five findings preview what it argues:

1. **Imitation is real, small, and lives on function words.** Across eight LLMs (Claude, GPT, and local open-weight models) prompted to imitate fifteen novelists, naming the author roughly triples envelope entry on the function-word layer (from a 10.7% unprompted base rate to 30.5%) — and still, roughly seven in ten imitation samples land outside the target author's own range of variation.
2. **The chassis does not move.** Style prompting transfers texture, but the function-word layer's movement toward the target is statistically indistinguishable from zero. A model's function-word distribution behaves less like a style it can wear than a substrate it generates from.
3. **The author's text beats the author's name.** Handing a model an author's actual opening pages (with no name attached) out-performs naming the author in every model-matched comparison we could run. One frontier model refused that condition in 20 of 20 attempts while explicitly offering style imitation instead — provider policy postures now shape what imitation research can even measure, and we report that asymmetry as a finding.
4. **Folk "AI tells" cannot tell novelists from machines.** We operationalized twelve popular tells — the em dash, "not X, but Y," rule-of-three triads, "delve," corporate jargon, hedging — and scored celebrated novelists against AI long-form fiction. The combined score is a coin flip (AUC ≈ 0.51). A threshold tuned to catch half the machine samples falsely flags about half the passages by Morrison, Ishiguro, and Pynchon. Four tells run *backwards* — the novelists use more exclamation marks, superlatives, and hedges than the machines do. The em dash, the best single tell, turns out to be a model-family variable, not an AI constant.
5. **Models are nobody's copy — and somebody anyway.** Unprompted model output lands outside every shelf author's envelope; it is, statistically, no one's prose. Yet the models identify as themselves at 97.8% (leave-one-out attribution over the same function-word features; chance is 12.5%). Each model is its own stable, identifiable voice that is not any human's voice.

The practical reading for writers follows from the layer split: the features the tell-lists name live in the texture layer — the layer that moves on request and that celebrated human prose occupies densely — so revising your prose to dodge them is sanding off craft to satisfy a checklist that can't tell Morrison from a machine. What actually carries authorship is the layer no checklist reaches, and the closer the final language is to literally yours, the more the result measures as yours.

## How the work was tested

The findings survived a process I care about as much as the results:

- **Self-commissioned adversarial review, twice.** I had the headline results red-teamed before any reviewer would see them. Two earlier headline claims turned out to be artifacts — one of length calibration, one of refusal composition. They were retracted and replaced with the corrected analysis, and the full adversarial record is committed verbatim alongside the paper.
- **Validation gates with honest failures.** The measurement space had to pass a battery of pre-specified gates (separation, attribution, permutation nulls, length sensitivity, a same-author positive control). Where a strict gate failed, it is reported as a failure, not relabeled.
- **Every claim evidence-traced.** Each result in the draft carries a pointer to a committed, machine-readable artifact; the numbers were frozen and audited against those artifacts before this preview was written.

## Status

- **Paper:** drafted, post-audit, targeting **CHR 2027** (Computational Humanities Research).
- **Companion repository:** a curated research release — the instrument, validation harness, public-domain replication shelf, and a CC0 corpus of ~1,000 AI-generated long-form samples across eight models and five prompting conditions — goes public with the preprint, expected summer 2026.
- **This page** will link both the moment they're up.

## Honest caveats

This is a single-researcher project under adversarial self-review, not yet a peer-reviewed result. The contemporary-shelf source texts are copyrighted novels and will never be redistributed; the replication path runs through the public-domain shelf and released aggregate artifacts. Scope is long-form fiction at document scale — claims here are not claims about chat text, student essays, or short passages, and the paper is explicit about where the measurement's validity ends.

I also ran the original instrument end-to-end on a full novel I wrote — the most *motivated* test I could give the method, which is precisely why the measurement had to be rebuilt on authors who are not me; the personal stake is what keeps the research honest: the goal was never "make AI signals go away," it was *I authored the final language; it reads like me.*

## Provenance and authorship

The questions Mnemosyne asks, its design, and the standard it's held to are mine; the implementation was built working alongside AI — directing LLM collaborators (Claude among them) through the code, then commissioning the same classes of models to attack the results. A study of machine imitation built in collaboration with the machines it studies has an obvious conflict to manage, which is why the methodology leans on pre-specified gates, multi-vendor models, adversarial review, and a replication path that doesn't require trusting me. The evidence doesn't care either way.

— William Cullen Bryant, 2025–present
