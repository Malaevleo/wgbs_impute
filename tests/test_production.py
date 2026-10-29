"""Portable regressions for data integrity and distribution boundaries."""

from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from wgbs_impute.cli import main
from wgbs_impute.docker_entry import (
    main as docker_main,
    render_dashboard,
    run_demo,
    run_pipeline,
    summarize,
    write_bed,
)
from wgbs_impute.impute import impute_query, parse_kappa_arg
from wgbs_impute.io import ParseError, read_manifest, read_sites, write_manifest
from wgbs_impute.panel import load_root, prepare_panel
from wgbs_impute.thinning import subsample_depths


def bed(path, chrom="chr1", pct=50, cov=80, sites=12):
    path.write_text(
        "".join(
            f"{chrom}\t{i * 100}\t{i * 100 + 1}\t.\t0\t+\t{i * 100}\t{i * 100 + 1}\t0\t{cov}\t{pct}\n"
            for i in range(sites)
        )
    )


def table(path):
    with path.open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


class ProductionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def panel(self, split=False):
        refs = self.root / "refs"
        refs.mkdir()
        for j, pct in enumerate((20, 40, 60, 80)):
            for ch in ("chr1", "chr2") if split else ("chr1",):
                bed(refs / f"sample{j}.{ch}.bed", ch, pct)
        manifest = self.root / "manifest.tsv"
        write_manifest(refs, manifest)
        return refs, manifest

    def test_demo_recovers_known_signal_above_panel_mean_baseline(self):
        out = self.root / "demo"
        self.assertEqual(run_demo(out), 0)
        rows = table(out / "imputed.tsv")
        self.assertEqual(len(rows), 400)
        self.assertGreaterEqual(sum(r["cov"] == "0" for r in rows), 200)
        self.assertTrue(all(0 <= float(r["pct"]) <= 100 for r in rows))
        report = json.loads((out / "imputed.tsv.run.json").read_text())
        self.assertNotIn("lib0", report["panel_sample_ids"])

    def test_split_chromosomes_preserve_identity_and_exclusion(self):
        _, manifest = self.panel(True)
        self.assertEqual(len(read_manifest(manifest)), 8)
        cache = self.root / "cache"
        prepare_panel(manifest, cache)
        self.assertEqual(load_root(cache)["chromosomes"], ["chr1", "chr2"])
        self.assertEqual(len(load_root(cache)["sample_ids"]), 4)
        query = self.root / "sample0.chr2.bed"
        bed(query, "chr2", 50, 2)
        report = impute_query(query, self.root / "out.tsv", panel_cache=cache, kappa=8)
        self.assertNotIn("sample0", report["panel_sample_ids"])
        self.assertEqual(len(report["panel_sample_ids_by_chromosome"]["chr2"]), 3)

    def test_disease_filter_applies_before_same_directory_files_enter(self):
        refs, _ = self.panel()
        (refs / "samples.tsv").write_text("epirr\tdisease\trole\nsample0\tA\tref\nsample1\tB\ttarget\n")
        rows = write_manifest(refs, self.root / "selected.tsv", disease_dir=refs)
        self.assertEqual([(r["sample_id"], r["group"]) for r in rows], [("sample0", "A")])
        rows = write_manifest(refs, self.root / "all.tsv", disease_dir=refs, disease_role="all")
        self.assertEqual({r["sample_id"] for r in rows}, {"sample0", "sample1"})
        (refs / "samples.tsv").write_text("epirr\tdisease\trole\n")
        with self.assertRaisesRegex(ParseError, "no library files"):
            write_manifest(refs, self.root / "empty.tsv", disease_dir=refs)

    def test_include_panel_sites_adds_absent_chromosomes_and_positions(self):
        _, manifest = self.panel(True)
        cache = self.root / "cache"
        prepare_panel(manifest, cache)
        query, output = self.root / "query.cov", self.root / "out.tsv.gz"
        query.write_text("chr1\t101\t101\t50\t1\t1\n")
        impute_query(query, output, panel_cache=cache, include_panel_sites=True)
        with gzip.open(output, "rt") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        self.assertEqual(len(rows), 24)
        self.assertEqual(sum(int(r["cov"]) > 0 for r in rows), 1)
        self.assertTrue(all(r["pct"] for r in rows))
        self.assertEqual({r["chrom"] for r in rows}, {"chr1", "chr2"})

    def test_atomic_output_preserves_previous_result_on_failure(self):
        from wgbs_impute.impute import _write_table

        output = self.root / "result.tsv"
        output.write_text("previous result\n")

        def interrupted_rows():
            yield ["first"]
            raise ValueError("interrupted")

        with self.assertRaisesRegex(ValueError, "interrupted"):
            _write_table(output, ["header"], interrupted_rows())
        self.assertEqual(output.read_text(), "previous result\n")
        self.assertEqual(list(self.root.iterdir()), [output])

    def test_output_cannot_overwrite_reference_or_cache(self):
        refs, manifest = self.panel()
        cache = self.root / "cache"
        prepare_panel(manifest, cache)
        query = self.root / "query.bed"
        bed(query, cov=2)
        for output in (query, refs / "sample0.chr1.bed", cache / "chr1" / "pos.npy"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "overwrite"):
                impute_query(query, output, panel_cache=cache)

    def test_corrupt_cached_chromosome_is_an_error_not_passthrough(self):
        from wgbs_impute.panel import load_chrom

        _, manifest = self.panel()
        cache = self.root / "cache"
        prepare_panel(manifest, cache)
        np.save(cache / "chr1" / "beta1.npy", np.zeros((2, 2)))
        with self.assertRaisesRegex(ValueError, "shapes"):
            load_chrom(cache, "chr1")

    def test_only_below_and_score_all_are_exclusive(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            impute_query(
                self.root / "q",
                self.root / "out",
                panel_cache=self.root / "cache",
                only_below=True,
                score_all=True,
            )

    def test_ambiguous_identity_raises(self):
        refs, _ = self.panel()
        bed(refs / "sample0_other.chr1.bed")
        with self.assertRaisesRegex(ParseError, "ambiguous"):
            write_manifest(refs, self.root / "ambiguous.tsv")

    def test_bismark_coordinates(self):
        for filename, line in (
            ("query.cov", "chr1\t101\t101\t50\t2\t2\n"),
            ("query.zero.cov", "chr1\t100\t101\t50\t2\t2\n"),
        ):
            path = self.root / filename
            path.write_text(line)
            got = read_sites(path)["chr1"]
            self.assertEqual((got["start"][0], got["end"][0]), (100, 101))

    def test_numeric_chromosomes_in_counts(self):
        path = self.root / "counts.tsv"
        path.write_text("1\t100\t101\t1\t2\n")
        self.assertEqual(read_sites(path, "counts")["1"]["cov"][0], 2)

    def test_invalid_rows_fail_with_line_numbers(self):
        for line in (
            "chr1\t-1\t1\t1\t2",
            "chr1\t1\t2\t3\t2",
            "chr1\t1\t2\tnan\t2",
            "chr1\t1\t2\t0\tinf",
            "chr1\t1\t1\t0\t0",
            "../bad\t1\t2\t1\t2",
            "chr1\t1\t2\t0.5\t2",
            "chr1\t1\t2\t0\t4294967296",
            "broken",
        ):
            with self.subTest(line=line):
                path = self.root / "bad.tsv"
                path.write_text(line + "\n")
                with self.assertRaisesRegex(ParseError, "bad.tsv:1:"):
                    read_sites(path, "counts")
        bed(path := self.root / "bad.bed", pct=101)
        with self.assertRaisesRegex(ParseError, "percent"):
            read_sites(path)

    def test_absent_chromosomes_preserved_and_missing_predictions_not_zero(self):
        _, manifest = self.panel()
        cache = self.root / "cache"
        prepare_panel(manifest, cache)
        query, output = self.root / "query.bed", self.root / "out.tsv"
        bed(query, "chrMissing", cov=2)
        impute_query(query, output, panel_cache=cache)
        self.assertEqual(len(table(output)), 12)
        self.assertTrue(all(r["source"] == "passthrough" for r in table(output)))
        rows = [
            {"chrom": "chr1", "start": "0", "end": "1", "cov": "0", "pct": "", "pct_obs": ""},
            {"chrom": "chr1", "start": "1", "end": "2", "cov": "0", "pct": "0.0000", "pct_obs": ""},
        ]
        dest = self.root / "out.bed.gz"
        write_bed(dest, rows, "pairwise", {})
        with gzip.open(dest, "rt") as fh:
            written = [line for line in fh if not line.startswith("#")]
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0].split("\t")[1], "1")

    def test_failed_preparation_never_publishes_cache(self):
        _, manifest = self.panel()
        cache = self.root / "cache"
        with patch("wgbs_impute.panel._prepare_chrom", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                prepare_panel(manifest, cache)
        self.assertFalse(cache.exists())
        self.assertFalse(cache.with_name("cache.lock").exists())
        prepare_panel(manifest, cache)
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare_panel(manifest, cache)

    def test_compact_cache_and_schema_validation(self):
        _, manifest = self.panel()
        cache = self.root / "cache"
        prepare_panel(manifest, cache)
        self.assertFalse((cache / "chr1" / "snn.npy").exists())
        meta = load_root(cache)
        meta["schema"] = -1
        (cache / "meta.json").write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "rebuild"):
            load_root(cache)

    def test_invalid_options_and_kappa(self):
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_kappa_arg(value)
        path = self.root / "kappa.tsv"
        path.write_text("depth kappa\n2 8\n2 10\n")
        with self.assertRaises(ValueError):
            parse_kappa_arg(str(path))
        _, manifest = self.panel()
        for options in ({"neighbors": -1}, {"neighbors": 2.5}, {"shrink": 2}, {"window": -1}, {"seed": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                prepare_panel(manifest, self.root / "invalid", **options)
        self.assertEqual(
            main(
                [
                    "prepare-panel",
                    "--manifest",
                    str(manifest),
                    "--cache",
                    str(self.root / "bad"),
                    "--neighbors",
                    "-1",
                ]
            ),
            2,
        )
        self.assertEqual(docker_main(["--neighbors", "-1"]), 2)

    def test_thinning_reproducible_and_preserves_subsets(self):
        meth, cov = np.array([0, 20, 40]), np.array([0, 40, 40])
        first = subsample_depths(meth, cov, [2, 5], 40, 42)
        second = subsample_depths(meth, cov, [2, 5], 40, 42)
        for depth, (m, c) in first.items():
            np.testing.assert_array_equal(m, second[depth][0])
            self.assertTrue(((0 <= m) & (m <= c) & (c <= cov) & (m <= meth)).all())
            self.assertTrue(((c - m) <= (cov - meth)).all())

    def test_container_outputs_and_missing_prediction_accounting(self):
        refs, _ = self.panel()
        query = self.root / "query.bed"
        bed(query, cov=2)
        with query.open("a") as fh:
            fh.write("chr1\t999999\t1000000\t.\t0\t+\t999999\t1000000\t0\t0\t0\n")
        out = self.root / "out"
        options = dict(
            preset="research",
            max_cov=10,
            only_below=False,
            chrom=None,
            n_refs=0,
            disease_role="ref",
            method="blup",
        )
        run_pipeline(refs, query, out, **options)
        for name in (
            "imputed.tsv",
            "imputed.bed.gz",
            "stats.tsv",
            "stats_by_coverage.tsv",
            "stats_by_sd.tsv",
            "dashboard.html",
            "missing_predictions.tsv",
            "imputed.tsv.run.json",
            "run.complete.json",
        ):
            self.assertTrue((out / name).is_file(), name)
        self.assertEqual(len(table(out / "missing_predictions.tsv")), 1)
        self.assertEqual(table(out / "stats.tsv")[-1]["n_missing_predictions"], "1")
        with self.assertRaisesRegex(ValueError, "empty"):
            run_pipeline(refs, query, out, **options)

    def test_html_escapes_chromosome_names(self):
        rows = [
            {
                "chrom": "<script>alert(1)</script>",
                "cov": "0",
                "pct": "50",
                "pct_obs": "",
                "panel_sd": "5",
                "source": "blup",
            }
        ]
        stats, cov, sd = summarize(rows, {"chromosomes": {}, "panel_sample_ids": []}, 10)
        page = render_dashboard(stats, cov, sd, "research")
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;", page)


if __name__ == "__main__":
    unittest.main()
