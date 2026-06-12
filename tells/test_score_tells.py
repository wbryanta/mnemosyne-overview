"""Tests for the folk-tell counters and statistics in score_tells.py.

The counters are conservative regex/heuristics; these tests pin their
behavior on constructed snippets (true positives counted, documented
exclusions not counted) and the AUC/threshold/percentile machinery on
known values. Adapted from the canonical analysis tool's test suite, with
the NumPy-dependent paths rewritten for the pure-stdlib module.

Run with either:
    python3 -m pytest test_score_tells.py
    python3 test_score_tells.py        (stdlib unittest fallback)
"""

import random
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import score_tells as st  # noqa: E402


def count(tell_id, text):
    counters = {tid: fn for tid, _, fn in st.TELLS}
    return counters[tell_id](text)


class TestCounters(unittest.TestCase):
    def test_em_dash_counts_both_forms(self):
        self.assertEqual(count("em_dash", "a — b and c -- d"), 2)

    def test_em_dash_ignores_triple_hyphen_rule(self):
        # standalone -- only; --- (markdown rule) is not two em dashes
        self.assertEqual(count("em_dash", "a --- b"), 0)

    def test_em_dash_ignores_en_dash_and_horizontal_bar(self):
        # en dash (U+2013) and horizontal bar (U+2015) are not em dashes
        self.assertEqual(count("em_dash", "pages 3–7, then ― a pause"), 0)

    def test_not_x_but_y_span_bound_40_chars(self):
        # the regex bounds the 'not ... , but' span at 40 chars; longer
        # spans (and anything crossing stop punctuation) do not match
        within = "It was not " + "x" * 30 + ", but rain."
        beyond = "It was not " + "x" * 45 + ", but rain."
        self.assertEqual(count("not_x_but_y", within), 1)
        self.assertEqual(count("not_x_but_y", beyond), 0)

    def test_not_x_but_y(self):
        self.assertEqual(count("not_x_but_y", "It was not anger, but grief."), 1)
        self.assertEqual(
            count("not_x_but_y", "It's not the fall, it's the landing."), 1)
        # 'not only ... but also' (classical rhetoric) excluded
        self.assertEqual(count("not_x_but_y", "not only smart, but also kind"), 0)
        # cross-sentence 'not ... but' not counted
        self.assertEqual(count("not_x_but_y", "He did not go. But she did."), 0)

    def test_tricolon_serial_triad(self):
        self.assertEqual(count("tricolon", "cold, dark, and wet"), 1)
        self.assertEqual(count("tricolon", "the night, the road, and rain"), 1)
        # two items only — not a triad
        self.assertEqual(count("tricolon", "cold and wet"), 0)
        # conservative: 3-word items are documented as missed
        self.assertEqual(
            count("tricolon",
                  "the cold dark night, the long wet road, and rain"), 0)

    def test_exclamation(self):
        self.assertEqual(count("exclamation", "No! Stop! Please."), 2)

    def test_lets_opener_sentence_initial_only(self):
        self.assertEqual(count("lets_opener", "Let's begin. Then we stop."), 1)
        self.assertEqual(count("lets_opener", "He said. Let's go now."), 1)
        # mid-sentence "let's" is not an opener
        self.assertEqual(count("lets_opener", "then let's go"), 0)

    def test_superlative(self):
        self.assertEqual(count("superlative", "the greatest, the most beautiful"), 2)
        self.assertEqual(count("superlative", "the best and the worst"), 2)
        # blocklisted -est words
        self.assertEqual(count("superlative", "an honest harvest in the forest"), 0)
        # bare 'most + noun' not counted (own tell territory)
        self.assertEqual(count("superlative", "most people think"), 0)

    def test_delve_leverage_lemmas(self):
        self.assertEqual(count("delve_leverage", "We delved in, leveraging it."), 2)
        self.assertEqual(count("delve_leverage", "the lever moved"), 0)

    def test_corporate_jargon(self):
        self.assertEqual(
            count("corporate_jargon",
                  "a scalable, holistic paradigm for stakeholders"), 4)
        self.assertEqual(count("corporate_jargon", "she walked to the store"), 0)

    def test_hedges_exclude_bare_modals(self):
        self.assertEqual(count("hedges", "Perhaps it was, arguably, enough."), 2)
        self.assertEqual(count("hedges", "He may go. She might stay."), 0)

    def test_staging_adverbs(self):
        self.assertEqual(count("staging_adverbs", "quietly devastating, softly lit"), 2)

    def test_container_words_framing_only(self):
        self.assertEqual(count("container_words", "a space for grief"), 1)
        self.assertEqual(count("container_words", "an opportunity to grow"), 1)
        # literal space not counted
        self.assertEqual(count("container_words", "the space between the houses"), 0)

    def test_unnamed_consensus(self):
        self.assertEqual(
            count("unnamed_consensus", "Most people agree; studies show it."), 2)


class TestStats(unittest.TestCase):
    def test_auc_perfect_separation(self):
        self.assertEqual(st.mann_whitney_auc([3.0, 4.0], [1.0, 2.0]), 1.0)

    def test_auc_chance_on_identical(self):
        self.assertAlmostEqual(
            st.mann_whitney_auc([1.0, 2.0], [1.0, 2.0]), 0.5, places=12)

    def test_auc_inverted(self):
        self.assertEqual(st.mann_whitney_auc([1.0, 2.0], [3.0, 4.0]), 0.0)

    def test_tell_stats_degenerate_when_ai_median_zero(self):
        rng = random.Random(0)
        ai = [0.0] * 6 + [1.0] * 4
        human = [abs(rng.gauss(1, 0.5)) for _ in range(40)]
        s = st.tell_stats(ai, human, ["a"] * 40)
        self.assertIs(s["witch_hunt"]["degenerate_threshold"], True)
        self.assertEqual(s["witch_hunt"]["most_flagged_authors"], [])

    def test_tell_stats_not_degenerate_for_signed_scores(self):
        rng = random.Random(0)
        ai = [rng.gauss(-1, 1) for _ in range(50)]   # negative median, signed
        human = [rng.gauss(0, 1) for _ in range(50)]
        s = st.tell_stats(ai, human, ["a"] * 50)
        self.assertIs(s["witch_hunt"]["degenerate_threshold"], False)

    def test_rates_per_1000_words(self):
        text = "a — b " * 10  # 10 em dashes, 30 words
        rates = st.rates_for(text, 30)
        self.assertAlmostEqual(rates["em_dash"], 10 * 1000 / 30, places=9)


class TestPercentileMachinery(unittest.TestCase):
    # Replaces the canonical suite's corpus-windowing tests (no corpus
    # loading here); pins the NumPy-compatible percentile/median math the
    # reproduction path depends on.

    def test_median_odd_and_even(self):
        self.assertEqual(st._median([3.0, 1.0, 2.0]), 2.0)
        self.assertEqual(st._median([4.0, 1.0, 2.0, 3.0]), 2.5)

    def test_percentile_matches_numpy_linear_interpolation(self):
        vals = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(st._percentile(vals, 25), 1.75, places=12)
        self.assertAlmostEqual(st._percentile(vals, 50), 2.5, places=12)
        self.assertAlmostEqual(st._percentile(vals, 75), 3.25, places=12)
        self.assertEqual(st._percentile(vals, 0), 1.0)
        self.assertEqual(st._percentile(vals, 100), 4.0)
        # interior point, t >= 0.5 lerp branch
        self.assertAlmostEqual(st._percentile([0.0, 10.0], 95), 9.5, places=12)


class TestCLI(unittest.TestCase):
    def test_reproduce_exits_zero_against_bundled_json(self):
        proc = subprocess.run(
            [sys.executable, str(HERE / "score_tells.py"), "--reproduce"],
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("RESULT: all", proc.stdout)
        self.assertNotIn("\nFAIL", proc.stdout)

    def test_scoring_path_on_constructed_snippet(self):
        # 20 words, 2 em dashes, 1 hedge -> em_dash 100/1k, hedges 50/1k
        words = ["alpha"] * 17 + ["—", "—", "perhaps"]
        text = " ".join(words)
        import json
        data = json.loads((HERE / "folk_tells_results.json").read_text(
            encoding="utf-8"))
        table = st.render_score_table("snippet", text, data)
        self.assertIn("em_dash", table)
        self.assertIn("100.00", table)   # em-dash rate per 1,000 words
        self.assertIn("50.00", table)    # hedge rate per 1,000 words
        self.assertIn("WARNING", table)  # < 1,000 words
        # human-novelist median for em_dash from the bundled aggregates
        self.assertIn("1.71", table)

    def test_scoring_cli_renders_table_for_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "snippet.txt"
            path.write_text(
                "It was not anger, but grief — cold, dark, and wet. " * 30,
                encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(HERE / "score_tells.py"), str(path)],
                capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("not_x_but_y", proc.stdout)
        self.assertIn("not a detector", proc.stdout)


if __name__ == "__main__":
    unittest.main()
