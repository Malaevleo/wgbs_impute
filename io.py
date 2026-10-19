"""Query and panel file readers. Coverage is never capped at 255."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

FORMATS = ("bedmethyl", "bismark_cov", "cache_tsv", "counts")


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
    if name.endswith(".cov") or ".cov." in name or name.endswith(".cov.gz"):
        return "bismark_cov"
    with open_text(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("track", "#")):
                continue
            n = len(line.rstrip("\n").split("\t"))
            if n >= 11:
                return "bedmethyl"
            if n == 6:
                return "bismark_cov"
            if n == 5:
                return "counts"
            if n == 3:
                return "cache_tsv" if chrom else "cache_tsv"
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
        for line in fh:
            if not line.strip() or line.startswith(("track", "#")):
                continue
            parts = line.rstrip("\n").split("\t")
            parsed = _parse_row(parts, fmt, chrom, path)
            if parsed is None:
                continue
            ch, start, end, meth, cov = parsed
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
    return out


def _parse_row(parts: list[str], fmt: str, chrom: str | None, path: Path):
    try:
        if fmt == "bedmethyl":
            if len(parts) < 11:
                return None
            ch, start, end = parts[0], int(parts[1]), int(parts[2])
            cov = float(parts[9])
            pct = float(parts[10])
            meth = _meth_from_pct(pct, cov)
        elif fmt == "bismark_cov":
            if len(parts) < 6:
                return None
            ch, start, end = parts[0], int(parts[1]), int(parts[2])
            meth = float(parts[4])
            unmeth = float(parts[5])
            cov = meth + unmeth
            meth = float(np.clip(meth, 0.0, cov))
        elif fmt == "cache_tsv":
            if len(parts) < 3:
                return None
            ch = chrom
            start = int(parts[0])
            cov = float(parts[1])
            pct = float(parts[2])
            end = start + 1
            meth = _meth_from_pct(pct, cov)
        elif fmt == "counts":
            if len(parts) >= 5 and not _is_int(parts[0]):
                ch, start, end = parts[0], int(parts[1]), int(parts[2])
                meth, cov = float(parts[3]), float(parts[4])
            elif len(parts) >= 3 and chrom:
                ch = chrom
                start = int(parts[0])
                end = start + 1
                meth, cov = float(parts[1]), float(parts[2])
            else:
                raise ParseError(f"{path}: counts rows need a chromosome column or --chrom")
            if cov < 0:
                return None
            meth = float(np.clip(meth, 0.0, cov))
        else:
            return None
    except ValueError:
        return None
    if cov < 0:
        return None
    return ch, start, end, meth, cov


def _is_int(text: str) -> bool:
    try:
        int(text)
    except ValueError:
        return False
    return True


def _meth_from_pct(pct: float, cov: float) -> float:
    if cov <= 0:
        return 0.0
    return float(np.clip(np.round(pct / 100.0 * cov), 0.0, cov))


def mean_positive_cov(cov: np.ndarray) -> float:
    pos = cov[cov > 0]
    return float(pos.mean()) if len(pos) else 0.0


def sample_id_from_filename(path: Path) -> str:
    """`IHECRE00000155_B_Lymphocyte.chr21.bed.gz` -> `IHECRE00000155`."""
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

    Healthy panels are BED files. Disease folders in this project are also
    `start cov pct` caches named `SAMPLE.chr21.tsv`, so those are included
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
        if name.endswith(".bed") or name.endswith(".bed.gz"):
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
                continue
            out[sid] = {
                "disease": (row.get("disease") or "").strip(),
                "role": (row.get("role") or "").strip(),
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

    files = [(path, "") for path in iter_panel_files(panel_dir)]
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
            if labels and role and disease_role != "all" and role != disease_role:
                continue
            files.append((path, info.get("disease") or Path(disease_dir).name))
    if not files:
        raise ParseError(f"{panel_dir}: no library files")
    output = Path(output)
    rows = []
    seen: set[str] = set()
    for path, group in files:
        sid = sample_id_from_filename(path)
        if sid in seen:
            print(f"warning: {sid} already in the manifest, skipping {path.name}", flush=True)
            continue
        seen.add(sid)
        try:
            rel = path.resolve().relative_to(output.parent.resolve())
            stored = str(rel)
        except ValueError:
            stored = str(path.resolve())
        rows.append({
            "path": stored,
            "sample_id": sid,
            "group": group,
            "chrom": chrom_from_filename(path),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["path", "sample_id", "group", "chrom"], delimiter="\t", lineterminator="\n",
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
                continue
            if sid in seen:
                raise ParseError(f"duplicate sample_id {sid}")
            seen.add(sid)
            sample_path = Path(rel)
            if not sample_path.is_absolute():
                candidate = path.parent / sample_path
                sample_path = candidate if candidate.exists() else Path(rel)
            rows.append({
                "path": sample_path,
                "sample_id": sid,
                "chrom": (row.get("chrom") or "").strip() or None,
                "group": (row.get("group") or "").strip(),
                "mean_cov": float(row["mean_cov"]) if (row.get("mean_cov") or "").strip() else None,
            })
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
