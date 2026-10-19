"""Tests for the imputation tool. They use synthetic files only."""

from __future__ import annotations

import gzip
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from wgbs_impute.blup import fuse, pairwise_shrunk_cov, select_neighbors
from wgbs_impute.cli import main
from wgbs_impute.io import read_sites
from wgbs_impute.optional import predict_comet, predict_global, predict_hmm, predict_kalman, predict_local


def _bed_line(chrom, start, meth, cov):
    pct = 0.0 if cov <= 0 else 100.0 * meth / cov
    cols = [chrom, str(start), str(start + 1), ".", "0", "+",
            str(start), str(start + 1), "0", str(cov), f"{pct:.6f}"]
    return "\t".join(cols)


def _write_bed(path: Path, rows):
    text = "\n".join(_bed_line(*r) for r in rows) + "\n"
    if str(path).endswith(".gz"):
        with gzip.open(path, "wt") as fh:
            fh.write(text)
    else:
        path.write_text(text)


class ParserTests(unittest.TestCase):
    def test_formats_keep_zero_and_large_counts(self):
        root = Path(self.id().split(".")[-1])
        # Isolated per-test directories live under the temp path unittest gives us via id only.
        # Write next to a TemporaryDirectory instead.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            bed = tmp / "q.bed.gz"
            _write_bed(bed, [("chr1", 10, 200, 400), ("chr1", 20, 0, 0), ("chr1", 30, 15, 10)])
            sites = read_sites(bed)["chr1"]
            self.assertEqual(list(sites["cov"]), [400, 0, 10])
            self.assertEqual(sites["meth"][0], 200)
            self.assertEqual(sites["meth"][2], 10)  # 150% would clip; 15 counts on cov 10 is already clipped by the writer
            covf = tmp / "q.cov.gz"
            covf.write_bytes(gzip.compress(
                b"chr1\t10\t11\t50\t5\t5\nchr1\t20\t21\t0\t0\t0\nchr1\t30\t31\t40\t300\t200\n"
            ))
            got = read_sites(covf)["chr1"]
            self.assertEqual(list(got["cov"]), [10, 0, 500])
            self.assertEqual(got["meth"][2], 300)
            cache = tmp / "q.tsv"
            cache.write_text("10\t400\t50\n20\t0\t0\n")
            got = read_sites(cache, "cache_tsv", chrom="chr21")["chr21"]
            self.assertEqual(list(got["start"]), [10, 20])
            self.assertEqual(got["meth"][0], 200)
            self.assertEqual(got["cov"][1], 0)

    def test_duplicate_keeps_higher_coverage(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.bed"
            _write_bed(path, [("chr1", 10, 1, 2), ("chr1", 10, 9, 10)])
            got = read_sites(path)["chr1"]
            self.assertEqual(len(got["start"]), 1)
            self.assertEqual(got["cov"][0], 10)
            self.assertEqual(got["meth"][0], 9)


class EstimatorTests(unittest.TestCase):
    def test_pairwise_moves_less_than_mean_fill(self):
        columns = np.column_stack([
            np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            np.array([0.0, 0.0, 0.0, 1.0, 1.0, np.nan]),
        ])
        _, cov = pairwise_shrunk_cov(columns, shrink=0.0)
        filled = columns.copy()
        for j in range(filled.shape[1]):
            mu = np.nanmean(filled[:, j])
            filled[~np.isfinite(filled[:, j]), j] = mu
        xc = filled - filled.mean(axis=0)
        meanfill = (xc.T @ xc) / (len(filled) - 1)
        self.assertGreater(abs(cov[0, 1]), abs(meanfill[0, 1]))

    def test_neighbors_stay_inside_the_window(self):
        pos = np.array([0, 100, 100_000])
        beta = np.array([[0.1, 0.2], [0.1, 0.2], [0.9, 0.1]])
        # Two references is below the usual overlap gate; lower it for the geometry check.
        neigh = select_neighbors(pos, beta, window=25000, k=5, min_overlap=2)
        self.assertIn(1, neigh[0])
        self.assertNotIn(2, neigh[0])
        self.assertTrue(np.all(np.abs(pos[neigh[0][neigh[0] >= 0]] - pos[0]) <= 25000))

    def test_fusion_edges(self):
        context = np.array([40.0])
        at_zero = fuse(context, np.array([0.0]), np.array([0.0]), 8.0)
        self.assertAlmostEqual(float(at_zero[0]), 40.0)
        high = fuse(np.array([0.0]), np.array([100.0]), np.array([100.0]), 8.0)
        self.assertLess(abs(float(high[0]) - 100.0), abs(0.0 - 100.0))


def _linked_panel(tmp: Path, n_sites=40, n_libs=12):
    """Every library is one methylation level, shared by all variable sites."""
    rng = np.random.default_rng(0)
    pos = np.arange(n_sites) * 100
    z = rng.uniform(0.15, 0.85, size=n_libs)
    beta = np.tile(z[:, None], (1, n_sites))
    beta[:, :4] = 0.2
    counts = np.full((n_libs, n_sites), 80, dtype=int)
    meth = np.rint(counts * beta).astype(int)
    return _write_panel(tmp, pos, meth, counts)


def _panel_and_query(tmp: Path, n_sites=80, n_libs=12, cov=40):
    rng = np.random.default_rng(0)
    pos = np.arange(n_sites) * 80
    z = rng.normal(size=n_libs)
    a = rng.uniform(-1.2, 1.2, size=n_sites)
    b = rng.normal(scale=1.4, size=n_sites)
    beta = 1.0 / (1.0 + np.exp(-(a[None, :] + np.outer(z, b))))
    counts = rng.poisson(cov, size=(n_libs, n_sites)) + 20
    meth = np.zeros_like(counts)
    for i in range(n_libs):
        meth[i] = rng.binomial(counts[i], np.clip(beta[i], 1e-3, 1 - 1e-3))
    # Eight sites are identical in every library, so the panel SD is 0.
    beta[:, :8] = 0.2
    counts[:, :8] = 100
    meth[:, :8] = 20
    return _write_panel(tmp, pos, meth, counts)


def _write_panel(tmp: Path, pos, meth, counts):
    n_libs, n_sites = meth.shape
    lib_dir = tmp / "libs"
    lib_dir.mkdir()
    ids = []
    for i in range(n_libs):
        sid = f"lib{i}"
        ids.append(sid)
        rows = [("chr1", int(pos[s]), int(meth[i, s]), int(counts[i, s])) for s in range(n_sites)]
        # A second chromosome with its own coordinates, so neighbor indices cannot leak.
        rows += [("chr2", int(pos[s]), int(meth[i, s]), int(counts[i, s])) for s in range(min(20, n_sites))]
        _write_bed(lib_dir / f"{sid}.bed.gz", rows)
    manifest = tmp / "panel.tsv"
    lines = ["path\tsample_id\tgroup"]
    for i, sid in enumerate(ids):
        group = "myeloid" if i < 4 else "T"
        lines.append(f"{lib_dir / (sid + '.bed.gz')}\t{sid}\t{group}")
    manifest.write_text("\n".join(lines) + "\n")
    return manifest, ids, pos, meth, counts


class EndToEndTests(unittest.TestCase):
    def test_query_is_removed_and_fused_beats_the_mean(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest, ids, pos, meth, counts = _linked_panel(tmp)
            cache = tmp / "cache"
            self.assertEqual(main(["prepare-panel", "--manifest", str(manifest), "--cache", str(cache),
                                   "--min-ref-mean-cov", "0"]), 0)
            # Corrupt the other chromosome. Imputing chr1 must not read it.
            (cache / "chr2" / "pos.npy").write_bytes(b"not a numpy file")
            query = tmp / "query.bed.gz"
            hold = 0
            # Even sites have no reads. Odd sites are observed deeply, so BLUP can
            # see which library this query resembles.
            q_rows = []
            for s in range(len(pos)):
                if s % 2 == 0:
                    q_rows.append(("chr1", int(pos[s]), 0, 0))
                else:
                    q_rows.append(("chr1", int(pos[s]), int(meth[hold, s]), int(counts[hold, s])))
            _write_bed(query, q_rows)
            out = tmp / "out.tsv"
            self.assertEqual(main([
                "impute", "--query", str(query), "--panel-cache", str(cache), "--output", str(out),
                "--sample-id", ids[hold], "--chrom", "chr1", "--threads", "1", "--kappa", "8",
            ]), 0)
            report = json.loads(Path(str(out) + ".run.json").read_text())
            self.assertNotIn(ids[hold], report["panel_sample_ids"])
            self.assertEqual(report["chromosomes"]["chr1"]["kappa"], 8.0)
            table = _read_out(out)
            self.assertTrue(all(r["chrom"] == "chr1" for r in table))
            low = [r for r in table if int(r["cov"]) < 10 and r["source"] not in ("passthrough", "panel_mean")]
            self.assertGreater(len(low), 10)
            truth = 100.0 * meth[hold] / counts[hold]
            err_f, err_m = [], []
            by_start = {int(pos[s]): truth[s] for s in range(len(pos))}
            for r in low:
                t = by_start[int(r["start"])]
                err_f.append(abs(float(r["pct"]) - t))
                err_m.append(abs(float(r["pct_panel"]) - t))
            self.assertLess(float(np.mean(err_f)), float(np.mean(err_m)))

    def test_low_sd_does_not_solve_and_only_below_drops_deep_rows(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest, ids, pos, meth, counts = _panel_and_query(tmp, n_sites=40, n_libs=10, cov=30)
            cache = tmp / "cache"
            main(["prepare-panel", "--manifest", str(manifest), "--cache", str(cache),
                  "--min-ref-mean-cov", "0"])
            query = tmp / "query.bed"
            # Coverage 2 keeps every site under the default threshold of 10.
            rows = [("chr1", int(pos[s]), 1, 2) for s in range(len(pos))]
            rows += [("chr1", int(pos[0]) + 5, 10, 20)]  # absent from the panel, and deep
            _write_bed(query, rows)
            out = tmp / "low.tsv"
            called = []
            real = __import__("wgbs_impute.blup", fromlist=["context_one"]).context_one

            def wrapped(*args, **kwargs):
                called.append(1)
                return real(*args, **kwargs)

            with patch("wgbs_impute.impute.context_one", side_effect=wrapped):
                main(["impute", "--query", str(query), "--panel-cache", str(cache),
                      "--output", str(out), "--sample-id", ids[0], "--chrom", "chr1",
                      "--kappa", "8", "--threads", "1"])
            table = _read_out(out)
            stable_rows = [r for r in table if r["panel_sd"] and float(r["panel_sd"]) < 5]
            self.assertGreater(len(stable_rows), 0)
            for r in stable_rows:
                self.assertEqual(r["source"], "panel_mean")
                self.assertAlmostEqual(float(r["pct"]), float(r["pct_panel"]), places=3)
                self.assertAlmostEqual(float(r["pct_blup"]), float(r["pct_panel"]), places=3)
            n_variable = sum(1 for r in table if r["panel_sd"] and float(r["panel_sd"]) >= 5
                             and r["source"] != "passthrough")
            self.assertEqual(len(called), n_variable)
            below = tmp / "below.tsv"
            main(["impute", "--query", str(query), "--panel-cache", str(cache),
                  "--output", str(below), "--sample-id", ids[0], "--chrom", "chr1",
                  "--only-below", "--kappa", "8"])
            self.assertTrue(_read_out(below))
            for r in _read_out(below):
                self.assertLess(int(r["cov"]), 10)

    def test_optional_methods_return_percents(self):
        rng = np.random.default_rng(1)
        n, r = 40, 8
        pos = np.arange(n) * 50
        beta = np.clip(rng.normal(0.6, 0.15, size=(n, r)), 0.05, 0.95)
        q_cov = np.full(n, 2.0)
        q_cov[:5] = 20
        q_meth = q_cov * beta[:, 0]
        hide = q_cov < 10
        panel_mean = beta.mean(axis=1) * 100
        panel_sd = beta.std(axis=1, ddof=1) * 100
        for pred in (
            predict_comet(pos, beta, q_meth, q_cov, panel_mean, hide),
            predict_hmm(pos, beta, q_meth, q_cov, hide),
            predict_kalman(pos, panel_mean, panel_sd, q_meth, q_cov, hide),
        ):
            self.assertEqual(len(pred), n)
            self.assertTrue(np.isfinite(pred).all())
            self.assertTrue(((pred >= 0) & (pred <= 100)).all())
        neigh = select_neighbors(pos, beta, 25000, 5, min_overlap=4)
        local = predict_local(beta, neigh, beta[:, 0], np.array([0, 1, 2]), iterations=20, depth=2, seed=1, min_ok=4)
        self.assertEqual(len(local), 3)
        self.assertTrue(np.isfinite(local).all())
        meth = np.rint(beta * 40)
        cov = np.full((n, r), 40.0)
        global_pred = predict_global(
            pos, meth, cov, beta, meth[:, 0], cov[:, 0], np.arange(n),
            max_train=2000, refnn_k=3, seed=1, threads=1, iterations=30,
        )
        self.assertEqual(len(global_pred), n)
        self.assertTrue(np.isfinite(global_pred).all())


def _read_out(path: Path):
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:] if line]


if __name__ == "__main__":
    unittest.main()
