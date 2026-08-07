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
import sensitivity_variants as sv  # noqa: E402


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


class TestSensitivityVariants(unittest.TestCase):
    # The subset/pooling sensitivity table published in README.md is
    # recomputed from the same bundled rows; these pin that it is a real
    # check and not a printout.

    def test_reproduce_exits_zero_against_bundled_json(self):
        proc = subprocess.run(
            [sys.executable, str(HERE / "sensitivity_variants.py"), "--reproduce"],
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("RESULT: all", proc.stdout)
        self.assertNotIn("\nFAIL", proc.stdout)

    def test_reproduce_fails_on_tampered_rows(self):
        import json
        import tempfile
        data = json.loads((HERE / "folk_tells_results.json").read_text(
            encoding="utf-8"))
        data["ai_sample_rows"][0]["em_dash"] += 5.0
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tampered.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(HERE / "sensitivity_variants.py"),
                 "--reproduce", "--data", str(path)],
                capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("FAIL", proc.stdout)

    def test_single_tell_z_sum_preserves_that_tells_auc(self):
        # z-scoring one tell is monotone, so pooling a single tell must
        # give exactly the AUC of its raw rates.
        human = [{"em_dash": v} for v in (0.5, 1.0, 2.0, 4.0)]
        ai = [{"em_dash": v} for v in (1.5, 3.0, 5.0, 6.0)]
        hz, az = sv.z_sum(human, ai, ["em_dash"])
        self.assertAlmostEqual(
            st.mann_whitney_auc(az, hz),
            st.mann_whitney_auc([r["em_dash"] for r in ai],
                                [r["em_dash"] for r in human]),
            places=12)

    def test_forward_subset_is_the_twelve_minus_the_four_backwards(self):
        self.assertEqual(len(sv.FORWARD_TELLS), 8)
        self.assertEqual(sorted(sv.FORWARD_TELLS + sv.BACKWARDS_TELLS),
                         sorted(st.TELL_IDS))


class TestExhaustiveSweep(unittest.TestCase):
    # The sweep is what turns "here are some subsets we tried" into "here
    # is the ceiling". These pin that it is exhaustive, that its fast path
    # agrees with the named-variant path, and that its maxima really do
    # bound every named variant of the same pooling.

    @classmethod
    def setUpClass(cls):
        import json
        data = json.loads((HERE / "folk_tells_results.json").read_text(
            encoding="utf-8"))
        cls.human = data["human_window_rows"]
        cls.ai = data["ai_sample_rows"]

    def test_sweep_covers_every_nonempty_subset(self):
        self.assertEqual(2 ** len(st.TELL_IDS) - 1, 4095)

    def test_component_path_matches_the_named_variant_path(self):
        # The exhaustive sweep sums precomputed per-tell components; the
        # named variants sum inside z_sum/rank_sum. Bit-identical, or the
        # sweep is measuring something other than what it reports.
        subsets = [["em_dash"], sv.BEST_PAIR, sv.HEADLINE_TELLS,
                   sv.FORWARD_TELLS, list(st.TELL_IDS)]
        human_log = sv.log1p_rows(self.human)
        ai_log = sv.log1p_rows(self.ai)
        for pooling, naive in (
                ("zsum_human",
                 lambda ts: sv.z_sum(self.human, self.ai, ts)),
                ("ranksum",
                 lambda ts: sv.rank_sum(self.human, self.ai, ts)),
                ("log1p_zsum",
                 lambda ts: sv.z_sum(human_log, ai_log, ts))):
            components = sv.pooling_components(self.human, self.ai, pooling)
            for tells in subsets:
                human, ai = naive(tells)
                self.assertEqual(
                    sv.subset_auc(components, tells),
                    st.mann_whitney_auc(ai, human),
                    "{0} disagrees on {1}".format(pooling, tells))

    def test_maxima_and_members_match_the_pinned_values(self):
        published = {key: value for key, _, value in sv.PUBLISHED_AUC}
        for key, pooling in sv.EXHAUSTIVE_VARIANTS:
            auc, members = sv.exhaustive_max(self.human, self.ai, pooling)
            self.assertAlmostEqual(auc, published[key], places=12)
            self.assertEqual(members, sv.PUBLISHED_ORACLE_MEMBERS[key])

    def test_oracle_maximum_bounds_every_named_variant_of_its_pooling(self):
        published = {key: value for key, _, value in sv.PUBLISHED_AUC}
        bounded = {
            "oracle_max_zsum": ["published", "forward8", "headline3", "pair2",
                                "named3", "em_dash_alone"],
            "oracle_max_ranksum": ["ranksum12", "ranksum8"],
            "oracle_max_log1p": ["log1p12", "log1p8"],
        }
        for oracle_key, keys in bounded.items():
            for key in keys:
                self.assertLessEqual(published[key], published[oracle_key],
                                     "{0} exceeds {1}".format(key, oracle_key))


class TestOracleClassification(unittest.TestCase):
    # Which variants an accuser could actually construct is a claim the
    # table makes in print; these pin it so it cannot drift silently. The
    # headline triple is label-free because it is the first three tells
    # this repository's own front page highlights -- not because it
    # happened to score well.

    def test_label_free_variants_are_exactly_the_three_documented_ones(self):
        all_twelve = ("published", "pooled12", "ranksum12", "log1p12")
        label_free = {key for key, label, _ in sv.PUBLISHED_AUC
                      if "*oracle*" not in label and key not in all_twelve}
        self.assertEqual(label_free, {"headline3", "named3", "em_dash_alone"})

    def test_every_exhaustive_variant_is_marked_oracle(self):
        labels = {key: label for key, label, _ in sv.PUBLISHED_AUC}
        for key, _pooling in sv.EXHAUSTIVE_VARIANTS:
            self.assertIn("*oracle*", labels[key])

    def test_headline_triple_is_the_readmes_first_three_tells(self):
        # The provenance claim, checked against the shipped README rather
        # than asserted in prose: the first tell phrase the page names is
        # the em dash, then "not X, but Y", then the rule of three -- which
        # is HEADLINE_TELLS, in order.
        text = (HERE / "README.md").read_text(encoding="utf-8").lower()
        opening = text[:text.index("## the numbers")]
        phrases = ["em dash", '"not x, but y', "rule-of-three triads"]
        for phrase in phrases:
            self.assertIn(phrase, opening)
        self.assertEqual(sorted(phrases, key=opening.index), phrases)
        self.assertEqual(sv.HEADLINE_TELLS,
                         ["em_dash", "not_x_but_y", "tricolon"])


class TestReadmeConsistency(unittest.TestCase):
    # --reproduce claims to verify the published tables. These pin that the
    # claim is real: the shipped READMEs pass, and a doctored one fails.

    def test_shipped_readmes_render_the_module_constants(self):
        failures = [msg for ok, msg in sv.check_readmes() if not ok]
        self.assertEqual(failures, [])

    def _tampered_copy(self, tmpdir, transform):
        import shutil
        tampered = Path(tmpdir) / "README.md"
        shutil.copy(sv.TELLS_README, tampered)
        tampered.write_text(
            transform(tampered.read_text(encoding="utf-8")), encoding="utf-8")
        return sv.check_readmes(
            auc_readmes=[(tampered, dict(sv.README_AUC_ROWS[0][1]))],
            any_rule_readme=tampered)

    def test_tampered_readme_value_fails(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            results = self._tampered_copy(
                tmpdir, lambda t: t.replace("| 0.5058 |", "| 0.6058 |"))
        failures = [msg for ok, msg in results if not ok]
        self.assertEqual(len(failures), 1, failures)
        self.assertIn("auc[published]", failures[0])

    def test_tampered_percentage_fails(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            results = self._tampered_copy(
                tmpdir, lambda t: t.replace("| 44.2% |", "| 40.2% |"))
        failures = [msg for ok, msg in results if not ok]
        self.assertEqual(len(failures), 1, failures)
        self.assertIn("any3[ai]", failures[0])

    def test_deleted_readme_row_fails(self):
        import tempfile

        def drop_row(text):
            return "\n".join(line for line in text.splitlines()
                             if not line.startswith("| em dash alone "))

        with tempfile.TemporaryDirectory() as tmpdir:
            results = self._tampered_copy(tmpdir, drop_row)
        failures = [msg for ok, msg in results if not ok]
        self.assertTrue(any("em_dash_alone" in msg for msg in failures),
                        failures)


if __name__ == "__main__":
    unittest.main()
