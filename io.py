"""Query and panel file readers. Coverage is never capped at 255."""

from __future__ import annotations

import gzip
from pathlib import Path

from wgbs_impute.validation import validate_chrom

import numpy as np

FORMATS = ("bedmethyl", "bismark_cov", "bismark_cov_zero", "cache_tsv", "counts")


class ParseError(ValueError):
    pass


def open_text(path: Path):
    raw = path.open("rb")
    magic = raw.read(2)
    raw.close()
    if magic == b"\x1f\x8b" or str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return path.open("rt")


def sniff_format(path: Path, chrom: str | None) -> str:
    name = path.name.lower()
    if name.endswith((".zero.cov", ".zero.cov.gz")):
        return "bismark_cov_zero"
    if name.endswith(".cov") or ".cov." in name or name.endswith(".cov.gz"):
        return "bismark_cov"
    with open_text(path) as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip() or line.startswith(("track", "browser", "#")):
                continue
            n = len(line.rstrip("\n").split("\t"))
            if n >= 11:
                return "bedmethyl"
            if n == 6:
                return "bismark_cov"
            if n == 5:
                return "counts"
            if n == 3:
                return "cache_tsv"
            raise ParseError(f"{path}: cannot detect format from {n} columns")
    raise ParseError(f"{path}: no data rows")


def read_sites(path: Path, fmt: str | None = None, chrom: str | None = None) -> dict[str, dict]:
    """Return chrom -> {start, end, meth, cov}, sorted and de-duplicated.

    Duplicate positions on one chromosome keep the row with the higher coverage.
    """
    path = Path(path)
    if fmt in (None, "auto"):
        fmt = sniff_format(path, chrom)
    if fmt not in FORMATS:
        raise ParseError(f"unknown format {fmt}")
    if fmt == "cache_tsv" and not chrom:
        raise ParseError(f"{path}: cache_tsv needs --chrom")
    buckets: dict[str, dict[int, tuple[int, float, float]]] = {}
    n_dup = 0
    with open_text(path) as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip() or line.startswith(("track", "browser", "#")):
                continue
            parts = line.rstrip("\n").split("\t")
            try:
                parsed = _parse_row(parts, fmt, chrom, path)
            except (ValueError, OverflowError) as exc:
                raise ParseError(f"{path}:{lineno}: {exc}") from exc
            ch, start, end, meth, cov = parsed
            if chrom and ch != chrom:
                continue
            slot = buckets.setdefault(ch, {})
            prev = slot.get(start)
            if prev is None or cov > prev[2]:
                if prev is not None:
                    n_dup += 1
                slot[start] = (end, meth, cov)
            else:
                n_dup += 1
    if n_dup:
        print(f"warning: {path.name}: kept the higher coverage at {n_dup} duplicate positions", flush=True)
    out = {}
    for ch, slot in buckets.items():
        starts = np.array(sorted(slot), dtype=np.int64)
        end = np.empty(len(starts), dtype=np.int64)
        meth = np.empty(len(starts), dtype=np.float64)
        cov = np.empty(len(starts), dtype=np.float64)
        for i, s in enumerate(starts):
            e, m, c = slot[int(s)]
            end[i], meth[i], cov[i] = e, m, c
        out[ch] = {"start": starts, "end": end, "meth": meth, "cov": cov}
    if not out and not chrom:
        raise ParseError(f"{path}: no data rows")
    return out


def _parse_row(parts: list[str], fmt: str, chrom: str | None, path: Path):
    if fmt == "bedmethyl":
        if len(parts) < 11:
            raise ParseError("bedMethyl requires at least 11 columns")
        ch, start, end = parts[0], int(parts[1]), int(parts[2])
        cov = float(parts[9])
        meth = 0.0 if cov == 0 and parts[10] in {".", "NA", ""} else _meth_from_pct(float(parts[10]), cov)
    elif fmt in {"bismark_cov", "bismark_cov_zero"}:
        if len(parts) != 6:
            raise ParseError("Bismark coverage requires 6 columns")
        ch, start, end = parts[0], int(parts[1]) - (fmt == "bismark_cov"), int(parts[2])
        _meth_from_pct(float(parts[3]), 1)
        meth, unmeth = float(parts[4]), float(parts[5])
        if not np.isfinite(unmeth) or unmeth < 0:
            raise ParseError("unmethylated count must be finite and nonnegative")
        cov = meth + unmeth
    elif fmt == "cache_tsv":
        if len(parts) != 3:
            raise ParseError("cache_tsv requires 3 columns")
        ch, start = chrom, int(parts[0])
        cov = float(parts[1])
        meth = _meth_from_pct(float(parts[2]), cov)
        end = start + 1
    elif fmt == "counts":
        if len(parts) == 5:
            ch, start, end = parts[0], int(parts[1]), int(parts[2])
            meth, cov = float(parts[3]), float(parts[4])
        elif len(parts) == 3 and chrom:
            ch, start = chrom, int(parts[0])
            end = start + 1
            meth, cov = float(parts[1]), float(parts[2])
        else:
            raise ParseError("counts needs 5 columns, or 3 columns with --chrom")
    else:
        raise ParseError(f"unknown format {fmt}")
    validate_chrom(ch)
    if start < 0 or end <= start or end >= np.iinfo(np.int64).max:
        raise ParseError("coordinates require 0 <= start < end")
    if not np.isfinite(cov) or not np.isfinite(meth) or not 0 <= meth <= cov:
        raise ParseError("counts require finite 0 <= methylated <= coverage")
    if cov != int(cov) or meth != int(meth):
        raise ParseError("read counts must be integers")
    if cov > np.iinfo(np.uint32).max:
        raise ParseError("coverage exceeds supported uint32 range")
    return ch, start, end, meth, cov


def _meth_from_pct(pct: float, cov: float) -> float:
    if not np.isfinite(pct) or not 0 <= pct <= 100:
        raise ParseError("methylation percent must be finite and between 0 and 100")
    return float(np.round(pct / 100.0 * cov))


def mean_positive_cov(cov: np.ndarray) -> float:
    pos = cov[cov > 0]
    return float(pos.mean()) if len(pos) else 0.0


def sample_id_from_filename(path: Path) -> str:
    """Infer sample identity from the filename; explicit manifests can override it."""
    name = path.name
    if name.endswith(".gz"):
        name = name[:-3]
    for suffix in (".cpg.bed", ".bed", ".cov", ".tsv"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    token = name.split("_", 1)[0].split(".", 1)[0]
    return token or name


def chrom_from_filename(path: Path) -> str:
    import re

    match = re.search(r"\.(chr(?:[0-9]+|X|Y|M))\.", path.name, re.IGNORECASE)
    return match.group(1) if match else ""


def iter_panel_files(panel_dir: Path, *, caches: bool = False) -> list[Path]:
    """Library files in one directory, not in subfolders.

    Panels accept BED and Bismark coverage files. Three-column
    `start cov pct` caches named `SAMPLE.chrN.tsv` are also included
    when `caches` is set. `samples.tsv` is a table, not a library.
    """
    panel_dir = Path(panel_dir)
    found = []
    for path in panel_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name.lower()
        if name in {"samples.tsv", "manifest.tsv"}:
            continue
        if name.endswith((".bed", ".bed.gz", ".cov", ".cov.gz")):
            found.append(path)
        elif caches and (name.endswith(".tsv") or name.endswith(".tsv.gz")):
            found.append(path)
    return sorted(found)


def read_disease_table(path: Path) -> dict[str, dict]:
    """Map sample id to disease and role. Accepts `epirr` or `sample_id`."""
    import csv

    out = {}
    with Path(path).open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if not reader.fieldnames:
            raise ParseError(f"{path}: disease table needs a header")
        id_col = "sample_id" if "sample_id" in reader.fieldnames else "epirr"
        if id_col not in reader.fieldnames or "disease" not in reader.fieldnames:
            raise ParseError(f"{path}: disease table needs disease and epirr or sample_id")
        for row in reader:
            sid = (row.get(id_col) or "").strip()
            if not sid:
                raise ParseError(f"{path}:{reader.line_num}: missing sample identity")
            if sid in out:
                raise ParseError(f"{path}:{reader.line_num}: duplicate disease sample {sid}")
            out[sid] = {
                "disease": (row.get("disease") or "").strip(),
                "role": (row.get("role") or "ref").strip(),
            }
    return out


def write_manifest(
    panel_dir: Path,
    output: Path,
    disease_dir: Path | None = None,
    disease_table: Path | None = None,
    disease_role: str = "ref",
) -> list[dict]:
    """Write path, sample_id, group, chrom.

    Healthy rows leave group blank. Disease rows set group to the disease
    name. With a table that has a role column, only `disease_role` is added
    unless that argument is `all`.
    """
    import csv

    same_dir = disease_dir is not None and Path(panel_dir).resolve() == Path(disease_dir).resolve()
    files = [] if same_dir else [(path, "") for path in iter_panel_files(panel_dir)]
    labels = {}
    if disease_dir is not None:
        table = Path(disease_table) if disease_table else None
        if table is None:
            for candidate in (Path(disease_dir) / "samples.tsv", Path(disease_dir).parent / "samples.tsv"):
                if candidate.exists():
                    table = candidate
                    break
        if table is not None:
            labels = read_disease_table(table)
        for path in iter_panel_files(disease_dir, caches=True):
            sid = sample_id_from_filename(path)
            info = labels.get(sid, {})
            role = info.get("role", "")
            if table is not None and (sid not in labels or (disease_role != "all" and role != disease_role)):
                continue
            files.append((path, info.get("disease") or Path(disease_dir).name))
    if not files:
        raise ParseError(f"{panel_dir}: no library files")
    output = Path(output)
    rows = []
    seen: set[tuple[str, str]] = set()
    for path, group in files:
        sid = sample_id_from_filename(path)
        key = (sid, chrom_from_filename(path))
        if key in seen:
            raise ParseError(
                f"ambiguous sample/chromosome {key}: use an explicit manifest with unique identities"
            )
        seen.add(key)
        try:
            rel = path.resolve().relative_to(output.parent.resolve())
            stored = str(rel)
        except ValueError:
            stored = str(path.resolve())
        rows.append(
            {
                "path": stored,
                "sample_id": sid,
                "group": group,
                "chrom": chrom_from_filename(path),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["path", "sample_id", "group", "chrom"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def read_manifest(path: Path) -> list[dict]:
    import csv

    path = Path(path)
    rows = []
    with path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if not reader.fieldnames or "path" not in reader.fieldnames or "sample_id" not in reader.fieldnames:
            raise ParseError(f"{path}: manifest needs header columns path and sample_id")
        seen = set()
        for row in reader:
            sid = (row.get("sample_id") or "").strip()
            rel = (row.get("path") or "").strip()
            if not sid or not rel:
                raise ParseError(f"{path}:{reader.line_num}: path and sample_id are required")
            ch = (row.get("chrom") or "").strip() or None
            if ch:
                validate_chrom(ch)
            key = (sid, ch)
            if key in seen:
                raise ParseError(f"duplicate sample_id/chromosome {key}")
            seen.add(key)
            sample_path = Path(rel)
            if not sample_path.is_absolute():
                sample_path = path.parent / sample_path
            rows.append(
                {
                    "path": sample_path,
                    "sample_id": sid,
                    "chrom": ch,
                    "group": (row.get("group") or "").strip(),
                    "mean_cov": float(row["mean_cov"]) if (row.get("mean_cov") or "").strip() else None,
                }
            )
    if not rows:
        raise ParseError(f"{path}: manifest has no libraries")
    return rows


def read_regions(path: Path) -> dict[str, list[tuple[int, int]]]:
    regions: dict[str, list[tuple[int, int]]] = {}
    with open_text(Path(path)) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track")):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            regions.setdefault(parts[0], []).append((int(parts[1]), int(parts[2])))
    for ch in regions:
        regions[ch].sort()
    return regions


def in_regions(chrom: str, start: int, regions: dict[str, list[tuple[int, int]]] | None) -> bool:
    if regions is None:
        return True
    spans = regions.get(chrom)
    if not spans:
        return False
    for a, b in spans:
        if a <= start < b:
            return True
        if a > start:
            break
    return False
