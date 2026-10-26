#!/usr/bin/env python3
"""Build a de-identified, PTB-XL-like dataset from vector ECG report PDFs.

The script deliberately creates two representations:

1. A provenance-preserving WFDB representation.  It places only waveform
   intervals that are visibly printed in the source PDF on a 10 s time grid
   and writes all unobserved intervals as WFDB missing samples (NaN).
2. A TRACE compatibility representation.  It periodically repeats the
   available within-lead segment to 10 s, cleans/resamples it at 250 Hz, and
   standardizes each lead.  This representation is convenient for the current
   TRACE input contract, but is not an original synchronous 12-lead ECG.

Original PDFs, source filenames, hospital identifiers, exact dates, names and
PDF metadata are never copied into the public output.  A keyed private mapping
is kept under --private-root and must never be released.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import stat
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import fitz
import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy.signal
import wfdb


LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
POINTS_PER_MV = 72.0 / 25.4 * 10.0
RECONSTRUCTION_FS = 500
TRACE_FS = 250
PUBLIC_VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SourceEntry:
    source_path: str
    relative_path: str
    source_sha256: str
    template: str
    ecg_id: int
    record_uid: str


class ExtractionError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument(
        "--ptbxl-root",
        type=Path,
        default=REPO_ROOT / "data",
        help=(
            "Directory containing the PTB-XL scp_statements.csv table "
            "(default: repository data directory)."
        ),
    )
    parser.add_argument(
        "--pilot-per-template",
        type=int,
        default=0,
        help="If positive, process this many files from each PDF layout subtype.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-trace-cleaning",
        action="store_true",
        help="Resample but do not call NeuroKit ECG cleaning in the TRACE layer.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def keyed_token(secret: bytes, namespace: str, value: str, length: int = 16) -> str:
    message = f"{namespace}:{value}".encode("utf-8")
    return hmac.new(secret, message, hashlib.sha256).hexdigest().upper()[:length]


def ensure_private_root(private_root: Path) -> bytes:
    private_root.mkdir(parents=True, exist_ok=True)
    os.chmod(private_root, stat.S_IRWXU)
    warning = private_root / "DO_NOT_RELEASE.md"
    if not warning.exists():
        warning.write_text(
            "# DO NOT RELEASE\n\n"
            "This directory contains the HMAC secret and source-to-public-ID mapping. "
            "It may contain hospital identifiers, exact source paths and dates. Keep it "
            "outside every public archive and repository.\n",
            encoding="utf-8",
        )
        os.chmod(warning, stat.S_IRUSR | stat.S_IWUSR)

    secret_path = private_root / "hmac_secret.hex"
    if secret_path.exists():
        secret = bytes.fromhex(secret_path.read_text(encoding="ascii").strip())
        if len(secret) < 32:
            raise RuntimeError("Existing HMAC secret is unexpectedly short.")
        return secret

    secret = secrets.token_bytes(32)
    secret_path.write_text(secret.hex() + "\n", encoding="ascii")
    os.chmod(secret_path, stat.S_IRUSR | stat.S_IWUSR)
    return secret


def classify_pdf(path: Path) -> tuple[str | None, dict[str, Any]]:
    with fitz.open(path) as doc:
        producer = str((doc.metadata or {}).get("producer") or "")
        pages = len(doc)
        rotation = int(doc[0].rotation) if pages else -1

    if "Aspose.Pdf" in producer and pages == 1 and rotation == 0:
        return "EDAN", {"pages": pages, "rotation": rotation, "producer_class": "Aspose"}
    if "Acrobat Distiller" in producer and pages == 1 and rotation == 90:
        return "PSCRIPT", {"pages": pages, "rotation": rotation, "producer_class": "Distiller"}
    if producer.startswith("iTextSharp") and pages == 3 and rotation == 90:
        return "PSCRIPT_BUNDLE", {"pages": pages, "rotation": rotation, "producer_class": "iText"}
    if "MUSE(R) Network Series" in producer and pages == 1 and rotation == 90:
        return "MUSE", {"pages": pages, "rotation": rotation, "producer_class": "MUSE"}
    return None, {"pages": pages, "rotation": rotation, "producer_class": "excluded"}


def inventory_sources(input_root: Path, private_root: Path, pilot_n: int) -> list[tuple[Path, str]]:
    paths = sorted(
        p
        for p in input_root.rglob("*.pdf")
        if "__MACOSX" not in p.parts and not p.name.startswith("._")
    )
    by_type: dict[str, list[Path]] = {"EDAN": [], "PSCRIPT": [], "PSCRIPT_BUNDLE": [], "MUSE": []}
    excluded_rows: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        template, info = classify_pdf(path)
        if template is None:
            excluded_rows.append(
                {
                    "source_relative_path": str(path.relative_to(input_root)),
                    "reason": "non_resting_ecg_or_unsupported_pdf_layout",
                    **info,
                }
            )
        else:
            by_type[template].append(path)
        if index % 250 == 0:
            print(f"[inventory] classified {index}/{len(paths)} PDFs", flush=True)

    if excluded_rows:
        excluded_path = private_root / "excluded_source_inventory.csv"
        pd.DataFrame(excluded_rows).to_csv(excluded_path, index=False)
        os.chmod(excluded_path, stat.S_IRUSR | stat.S_IWUSR)

    print("[inventory] resting ECG layout counts:", {k: len(v) for k, v in by_type.items()}, flush=True)
    print(f"[inventory] excluded reports: {len(excluded_rows)}", flush=True)

    selected: list[tuple[Path, str]] = []
    for template, template_paths in by_type.items():
        take = template_paths[:pilot_n] if pilot_n > 0 else template_paths
        selected.extend((path, template) for path in take)
    return selected


def build_source_entries(
    selected: Sequence[tuple[Path, str]], input_root: Path, secret: bytes
) -> list[SourceEntry]:
    staged: list[dict[str, str]] = []
    for index, (path, template) in enumerate(selected, start=1):
        digest = sha256_file(path)
        staged.append(
            {
                "source_path": str(path),
                "relative_path": str(path.relative_to(input_root)),
                "source_sha256": digest,
                "template": template,
                "record_uid": "AZE" + keyed_token(secret, "record", digest),
            }
        )
        if index % 100 == 0 or index == len(selected):
            print(f"[inventory] hashed {index}/{len(selected)} selected PDFs", flush=True)

    staged.sort(key=lambda row: row["record_uid"])
    return [SourceEntry(ecg_id=i, **row) for i, row in enumerate(staged, start=1)]


def point_xy(point: Any) -> tuple[float, float]:
    return float(point.x), float(point.y)


def drawing_polyline(drawing: dict[str, Any], cubic_steps: int = 20) -> np.ndarray:
    points: list[tuple[float, float]] = []
    for item in drawing.get("items", []):
        kind = item[0]
        if kind == "l":
            p0 = np.asarray(point_xy(item[1]), dtype=float)
            p1 = np.asarray(point_xy(item[2]), dtype=float)
            if not points or np.linalg.norm(np.asarray(points[-1]) - p0) > 1e-7:
                points.append((float(p0[0]), float(p0[1])))
            points.append((float(p1[0]), float(p1[1])))
        elif kind == "c":
            control = [np.asarray(point_xy(item[j]), dtype=float) for j in range(1, 5)]
            for t in np.linspace(0.0, 1.0, cubic_steps + 1):
                q = (
                    (1 - t) ** 3 * control[0]
                    + 3 * (1 - t) ** 2 * t * control[1]
                    + 3 * (1 - t) * t**2 * control[2]
                    + t**3 * control[3]
                )
                if not points or np.linalg.norm(np.asarray(points[-1]) - q) > 1e-7:
                    points.append((float(q[0]), float(q[1])))
    if len(points) < 2:
        raise ExtractionError("Waveform drawing has fewer than two usable points.")
    return np.asarray(points, dtype=np.float64)


def candidate_drawings(page: fitz.Page, template: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    drawings = page.get_drawings()
    standard: list[dict[str, Any]] = []
    rhythm: list[dict[str, Any]] = []
    for drawing in drawings:
        items = drawing.get("items", [])
        rect = drawing.get("rect")
        if rect is None or not items:
            continue
        kinds = {item[0] for item in items}
        if not kinds.issubset({"l", "c"}):
            continue
        if template == "EDAN":
            time_span = float(rect.x1 - rect.x0)
            if 300.0 <= time_span <= 430.0 and len(items) >= 1500:
                standard.append(drawing)
            elif time_span >= 600.0 and len(items) >= 3000:
                rhythm.append(drawing)
        elif template in {"PSCRIPT", "PSCRIPT_BUNDLE"}:
            time_span = float(rect.y1 - rect.y0)
            amp_span = float(rect.x1 - rect.x0)
            # High-amplitude/pathological leads can cross several printed row
            # bands; their bounding boxes legitimately exceed 130 PDF points.
            # Time span, stroke type and item count distinguish them from the
            # page/grid primitives, so retain amplitudes up to 300 points.
            if 300.0 <= time_span <= 450.0 and amp_span <= 300.0 and len(items) >= 100:
                standard.append(drawing)
            elif time_span >= 650.0 and amp_span <= 300.0 and len(items) >= 900:
                rhythm.append(drawing)
        elif template == "MUSE":
            time_span = float(rect.y1 - rect.y0)
            amp_span = float(rect.x1 - rect.x0)
            if 145.0 <= time_span <= 210.0 and amp_span <= 140.0 and len(items) >= 800:
                standard.append(drawing)
            elif time_span >= 650.0 and amp_span <= 140.0 and len(items) >= 3000:
                rhythm.append(drawing)
    return standard, rhythm


def robust_baseline(points: np.ndarray, amplitude_axis: int) -> float:
    values = points[:, amplitude_axis]
    lo, hi = np.quantile(values, [0.05, 0.95])
    central = values[(values >= lo) & (values <= hi)]
    return float(np.median(central if central.size else values))


def drawing_coordinates(
    drawing: dict[str, Any], template: str
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    points = drawing_polyline(drawing)
    if template == "EDAN":
        time_coord = points[:, 0]
        amplitude_coord = points[:, 1]
        baseline = robust_baseline(points, amplitude_axis=1)
    else:
        time_coord = -points[:, 1]
        amplitude_coord = points[:, 0]
        baseline = robust_baseline(points, amplitude_axis=0)

    order = np.argsort(time_coord, kind="mergesort")
    time_coord = time_coord[order]
    amplitude_coord = amplitude_coord[order]
    unique_time, inverse = np.unique(time_coord, return_inverse=True)
    if unique_time.size < 2:
        raise ExtractionError("Waveform drawing has no usable time span.")
    sums = np.bincount(inverse, weights=amplitude_coord)
    counts = np.bincount(inverse)
    amplitude_unique = sums / counts
    millivolts = -(amplitude_unique - baseline) / POINTS_PER_MV
    info = {
        "source_points": float(points.shape[0]),
        "time_span_pdf_points": float(unique_time[-1] - unique_time[0]),
        "baseline_pdf_point": baseline,
        "peak_abs_mv": float(np.max(np.abs(millivolts))),
        "std_mv": float(np.std(millivolts)),
    }
    return unique_time, millivolts.astype(np.float64), info


def resample_drawing(
    drawing: dict[str, Any], template: str, samples: int
) -> tuple[np.ndarray, dict[str, float]]:
    time_coord, millivolts, info = drawing_coordinates(drawing, template)
    target = np.linspace(float(time_coord[0]), float(time_coord[-1]), samples)
    signal = np.interp(target, time_coord, millivolts)
    return signal.astype(np.float32), info


def resample_synchronous_group(
    drawings: dict[str, dict[str, Any]], template: str, samples: int
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]], np.ndarray]:
    """Interpolate simultaneously printed leads on one shared PDF-time grid."""
    coordinates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for lead, drawing in drawings.items():
        time_coord, millivolts, info = drawing_coordinates(drawing, template)
        coordinates[lead] = (time_coord, millivolts)
        metrics[lead] = info
    common_start = max(value[0][0] for value in coordinates.values())
    common_end = min(value[0][-1] for value in coordinates.values())
    if common_end <= common_start:
        raise ExtractionError("Simultaneous lead group has no common PDF-time interval.")
    target = np.linspace(float(common_start), float(common_end), samples)
    signals = {
        lead: np.interp(target, time_coord, millivolts).astype(np.float32)
        for lead, (time_coord, millivolts) in coordinates.items()
    }
    return signals, metrics, target


def drawing_overlap_correlation(
    first: dict[str, Any], second: dict[str, Any], template: str
) -> float | None:
    time_a, signal_a, _ = drawing_coordinates(first, template)
    time_b, signal_b, _ = drawing_coordinates(second, template)
    common_start = max(float(time_a[0]), float(time_b[0]))
    common_end = min(float(time_a[-1]), float(time_b[-1]))
    if common_end <= common_start:
        return None
    points = max(250, int(round((common_end - common_start) / (common_end - common_start) * 2500)))
    target = np.linspace(common_start, common_end, points)
    first_signal = np.interp(target, time_a, signal_a)
    second_signal = np.interp(target, time_b, signal_b)
    if np.std(first_signal) <= 1e-8 or np.std(second_signal) <= 1e-8:
        return None
    return float(np.corrcoef(first_signal, second_signal)[0, 1])


def baseline_coordinate(drawing: dict[str, Any], template: str) -> float:
    points = drawing_polyline(drawing, cubic_steps=5)
    return robust_baseline(points, amplitude_axis=1 if template == "EDAN" else 0)


def split_evenly_by_time_center(
    drawings: Sequence[dict[str, Any]], groups: int, axis: str, reverse: bool
) -> list[list[dict[str, Any]]]:
    if len(drawings) != groups * 3 and not (groups == 2 and len(drawings) == 12):
        raise ExtractionError(f"Cannot divide {len(drawings)} standard paths into {groups} groups.")
    key = (
        (lambda d: float((d["rect"].x0 + d["rect"].x1) / 2.0))
        if axis == "x"
        else (lambda d: float((d["rect"].y0 + d["rect"].y1) / 2.0))
    )
    ordered = sorted(drawings, key=key, reverse=reverse)
    size = len(drawings) // groups
    return [ordered[i * size : (i + 1) * size] for i in range(groups)]


def map_paths_to_leads(
    standard: Sequence[dict[str, Any]], rhythm: Sequence[dict[str, Any]], template: str
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if len(standard) != 12:
        raise ExtractionError(f"Expected 12 standard waveform paths, found {len(standard)}.")

    lead_paths: dict[str, dict[str, Any]] = {}
    rhythm_paths: dict[str, dict[str, Any]] = {}
    if template == "EDAN":
        groups = split_evenly_by_time_center(standard, groups=2, axis="x", reverse=False)
        for group, names in zip(groups, (LEADS[:6], LEADS[6:])):
            ordered = sorted(group, key=lambda d: baseline_coordinate(d, template))
            lead_paths.update(dict(zip(names, ordered)))
        if len(rhythm) == 1:
            rhythm_paths["II"] = rhythm[0]
        elif len(rhythm) not in {0, 1}:
            raise ExtractionError(f"Expected zero or one EDAN rhythm path, found {len(rhythm)}.")
    elif template in {"PSCRIPT", "PSCRIPT_BUNDLE"}:
        groups = split_evenly_by_time_center(standard, groups=2, axis="y", reverse=True)
        for group, names in zip(groups, (LEADS[:6], LEADS[6:])):
            ordered = sorted(group, key=lambda d: baseline_coordinate(d, template))
            lead_paths.update(dict(zip(names, ordered)))
        if len(rhythm) != 1:
            raise ExtractionError(f"Expected one PScript rhythm path, found {len(rhythm)}.")
        rhythm_paths["II"] = rhythm[0]
    elif template == "MUSE":
        groups = split_evenly_by_time_center(standard, groups=4, axis="y", reverse=True)
        names_by_group = (LEADS[0:3], LEADS[3:6], LEADS[6:9], LEADS[9:12])
        for group, names in zip(groups, names_by_group):
            ordered = sorted(group, key=lambda d: baseline_coordinate(d, template))
            lead_paths.update(dict(zip(names, ordered)))
        if len(rhythm) != 3:
            raise ExtractionError(f"Expected three MUSE rhythm paths, found {len(rhythm)}.")
        ordered_rhythm = sorted(rhythm, key=lambda d: baseline_coordinate(d, template))
        rhythm_paths.update(dict(zip(("V1", "II", "V5"), ordered_rhythm)))
    else:
        raise ExtractionError(f"Unsupported layout {template}.")
    return lead_paths, rhythm_paths


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def canonical_100hz(signal500: np.ndarray, mask500: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    output = np.full((1000, len(LEADS)), np.nan, dtype=np.float32)
    mask100 = np.zeros((1000, len(LEADS)), dtype=bool)
    for lead_index in range(len(LEADS)):
        for start, end in contiguous_runs(mask500[:, lead_index]):
            segment = signal500[start:end, lead_index]
            target_len = int(round((end - start) / 5.0))
            down = scipy.signal.resample_poly(segment, 1, 5).astype(np.float32)
            if down.size != target_len:
                down = scipy.signal.resample(down, target_len).astype(np.float32)
            out_start = int(round(start / 5.0))
            out_end = min(1000, out_start + target_len)
            output[out_start:out_end, lead_index] = down[: out_end - out_start]
            mask100[out_start:out_end, lead_index] = True
    return output, mask100


def extract_waveforms(
    page: fitz.Page, template: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    standard, rhythm = candidate_drawings(page, template)
    lead_paths, rhythm_paths = map_paths_to_leads(standard, rhythm, template)
    signal = np.full((5000, len(LEADS)), np.nan, dtype=np.float32)
    mask = np.zeros((5000, len(LEADS)), dtype=bool)
    standard_segments: dict[str, np.ndarray] = {}
    path_qc: dict[str, Any] = {}

    if template in {"EDAN", "PSCRIPT", "PSCRIPT_BUNDLE"}:
        starts = {lead: (0 if index < 6 else 2500) for index, lead in enumerate(LEADS)}
        standard_samples = 2500
    else:
        starts = {lead: (index // 3) * 1250 for index, lead in enumerate(LEADS)}
        standard_samples = 1250

    if template in {"EDAN", "PSCRIPT", "PSCRIPT_BUNDLE"}:
        lead_groups = (LEADS[:6], LEADS[6:])
    else:
        lead_groups = (LEADS[0:3], LEADS[3:6], LEADS[6:9], LEADS[9:12])
    group_time_grids: list[np.ndarray] = []
    for lead_group in lead_groups:
        group_paths = {lead: lead_paths[lead] for lead in lead_group}
        group_signals, group_metrics, group_time = resample_synchronous_group(
            group_paths, template, standard_samples
        )
        group_time_grids.append(group_time)
        for lead in lead_group:
            lead_index = LEADS.index(lead)
            segment = group_signals[lead]
            start = starts[lead]
            end = start + standard_samples
            signal[start:end, lead_index] = segment
            mask[start:end, lead_index] = True
            standard_segments[lead] = segment
            path_qc[f"standard_{lead}"] = group_metrics[lead]

    rhythm_segments: dict[str, np.ndarray] = {}
    for lead, drawing in rhythm_paths.items():
        rhythm_time, rhythm_mv, info = drawing_coordinates(drawing, template)
        # Use the standard lead-column time grids rather than independently
        # stretching the long rhythm strip.  This preserves sample-level
        # alignment wherever a short standard lead overlaps the rhythm strip.
        segment = np.concatenate(
            [np.interp(group_time, rhythm_time, rhythm_mv) for group_time in group_time_grids]
        ).astype(np.float32)
        if segment.size != 5000:
            raise ExtractionError(
                f"Aligned rhythm lead {lead} has {segment.size} samples instead of 5000."
            )
        lead_index = LEADS.index(lead)
        signal[:, lead_index] = segment
        mask[:, lead_index] = True
        rhythm_segments[lead] = segment
        path_qc[f"rhythm_{lead}"] = info

    overlap_corr: dict[str, float | None] = {}
    for lead in rhythm_segments:
        overlap_corr[lead] = drawing_overlap_correlation(
            lead_paths[lead], rhythm_paths[lead], template
        )

    qc = {
        "template": "PSCRIPT" if template == "PSCRIPT_BUNDLE" else template,
        "layout_subtype": template,
        "standard_path_count": len(standard),
        "rhythm_path_count": len(rhythm),
        "observed_fraction": float(mask.mean()),
        "overlap_correlation": overlap_corr,
        "path_metrics": path_qc,
    }
    return signal, mask, qc, standard_segments


SENSITIVE_LINE_TERMS = (
    "姓名",
    "患者编号",
    "病历号",
    "登记号",
    "检查号",
    "申请号",
    "住院号",
    "门诊号",
    "医师",
    "医生",
    "审核",
    "检查日期",
    "报告日期",
    "打印日期",
    "科室",
    "床号",
)


def clean_diagnosis_lines(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).replace('"', "")
    text = re.sub(r"诊断(?:结果|结论|信息)\s*[:：]?", "", text)
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip(" \t;；,，。")
        if not line:
            continue
        if "仅供临床医生参考" in line:
            break
        if any(term in line for term in SENSITIVE_LINE_TERMS):
            continue
        line = re.sub(r"^\s*\d{1,2}\s*[、.．)）]\s*", "", line)
        line = re.sub(r"(?<!\d)\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}(?:日)?(?!\d)", "[DATE_REMOVED]", line)
        line = re.sub(r"(?<!\d)\d{7,}(?!\d)", "[ID_REMOVED]", line)
        line = line.strip(" \t;；,，。")
        if line and "[DATE_REMOVED]" not in line and "[ID_REMOVED]" not in line:
            lines.append(line)
    deduplicated: list[str] = []
    for line in lines:
        if line not in deduplicated:
            deduplicated.append(line)
    return deduplicated


def extract_diagnosis(page: fitz.Page, template: str) -> str:
    blocks = page.get_text("blocks", sort=False)
    if template == "EDAN":
        for block in blocks:
            text = str(block[4])
            if "诊断结果" in text or "诊断信息" in text:
                return "；".join(clean_diagnosis_lines(text))
        return ""

    for index, block in enumerate(blocks):
        text = str(block[4])
        if "诊断结论" not in text:
            continue
        same_block = clean_diagnosis_lines(text)
        if same_block:
            return "；".join(same_block)
        for candidate in blocks[index + 1 : index + 6]:
            candidate_text = str(candidate[4])
            candidate_lines = clean_diagnosis_lines(candidate_text)
            if candidate_lines and len(candidate_text) <= 300:
                joined = "；".join(candidate_lines)
                if any(
                    term in joined
                    for term in (
                        "心律",
                        "心电图",
                        "传导",
                        "阻滞",
                        "早搏",
                        "颤动",
                        "扑动",
                        "ST",
                        "T波",
                        "Q波",
                        "肥厚",
                        "梗死",
                        "缺血",
                    )
                ):
                    return joined
    return ""


def normalized_page_text(page: fitz.Page) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", page.get_text("text")))


def extract_demographics_and_hospital_id(page: fitz.Page, title: str) -> tuple[int | None, int | None, str, str]:
    text = normalized_page_text(page)
    age_match = re.search(r"年龄\s*[:：]?\s*(\d{1,3})\s*岁?", text)
    if age_match is None:
        age_match = re.search(r"[（(]\s*(\d{1,3})\s*岁\s*[）)]", text)
    age_raw = int(age_match.group(1)) if age_match else None
    age = min(age_raw, 90) if age_raw is not None and 0 <= age_raw <= 130 else None

    sex_match = re.search(r"性别\s*[:：]?\s*([男女])", text)
    if sex_match is None:
        sex_match = re.search(r"(?<![性男女性])([男女])性(?!别)", text)
    sex_char = sex_match.group(1) if sex_match else ""
    sex = 1 if sex_char == "男" else (0 if sex_char == "女" else None)

    hospital_id = ""
    for pattern in (
        r"(?:患者编号|病历号|登记号|ID)\s*[:：]?\s*([A-Za-z0-9_-]{6,20})",
        r"(?<!\d)(\d{10})(?!\d)",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            hospital_id = match.group(1)
            break
    if not hospital_id:
        title_ids = re.findall(r"(?<!\d)(\d{10})(?!\d)", title or "")
        if title_ids:
            hospital_id = title_ids[-1]
    patient_source = "hospital_identifier" if hospital_id else "record_fallback"
    return age, sex, hospital_id, patient_source


RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"窦性心动过速", ("STACH",)),
    (r"窦性心动过缓", ("SBRAD",)),
    (r"窦性心律不齐|窦性心律失常", ("SARRH",)),
    (r"窦性心律", ("SR",)),
    (r"心房颤动|房颤", ("AFIB",)),
    (r"心房扑动|房扑", ("AFLT",)),
    (r"阵发性室上性心动过速", ("PSVT",)),
    (r"室上性心动过速", ("SVTAC",)),
    (r"室性早搏|室早", ("PVC",)),
    (r"房性早搏|房早", ("PAC",)),
    (r"完全性右束支传导阻滞", ("CRBBB",)),
    (r"不完全性右束支传导阻滞", ("IRBBB",)),
    (r"完全性左束支传导阻滞", ("CLBBB",)),
    (r"不完全性左束支传导阻滞", ("ILBBB",)),
    (r"左前分支传导阻滞|左前半阻滞", ("LAFB",)),
    (r"左后分支传导阻滞|左后半阻滞", ("LPFB",)),
    (r"(?:一|I|1)度房室传导阻滞", ("1AVB",)),
    (r"(?:二|II|2)度房室传导阻滞", ("2AVB",)),
    (r"(?:三|III|3)度房室传导阻滞|完全性房室传导阻滞", ("3AVB",)),
    (r"室内传导阻滞|室内传导延迟", ("IVCD",)),
    (r"预激综合征|WPW", ("WPW",)),
    (r"左心室肥厚", ("LVH",)),
    (r"右心室肥厚", ("RVH",)),
    (r"左心房(?:增大|扩大|肥大)", ("LAO/LAE",)),
    (r"右心房(?:增大|扩大|肥大)", ("RAO/RAE",)),
    (r"下壁.*心肌梗死|心肌梗死.*下壁", ("IMI",)),
    (r"前间壁.*心肌梗死|心肌梗死.*前间壁", ("ASMI",)),
    (r"前壁.*心肌梗死|心肌梗死.*前壁", ("AMI",)),
    (r"侧壁.*心肌梗死|心肌梗死.*侧壁", ("LMI",)),
    (r"后壁.*心肌梗死|心肌梗死.*后壁", ("PMI",)),
    (r"心肌缺血", ("ISC_",)),
    (r"ST[-–— ]?T改变|ST段改变", ("NST_",)),
    (r"ST(?:段)?压低|ST下降", ("NST_", "STD_")),
    (r"ST(?:段)?抬高", ("STE_",)),
    (r"T波低平", ("NDT", "LOWT")),
    (r"T波倒置", ("NDT", "INVT")),
    (r"T波异常", ("NDT", "TAB_")),
    (r"QT间期延长|长QT", ("LNGQT",)),
    (r"低电压|低QRS电压", ("LVOLT",)),
    (r"异常Q波", ("QWAVE",)),
    (r"T波改变", ("NDT", "TAB_")),
    (r"正常心电图|大致正常心电图|正常范围心电图", ("NORM",)),
)


def map_scp_codes(diagnosis: str) -> tuple[dict[str, float], str]:
    codes: dict[str, float] = {}
    for pattern, matches in RULES:
        if re.search(pattern, diagnosis, flags=re.IGNORECASE):
            for code in matches:
                codes[code] = 100.0
    if not diagnosis:
        status = "no_diagnosis_text"
    elif not codes:
        status = "unmapped"
    else:
        status = "rule_based_unverified"
    return codes, status


def trace_compatible(
    canonical500: np.ndarray, mask500: np.ndarray, clean: bool
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    trace = np.zeros((len(LEADS), 2500), dtype=np.float32)
    valid_mask = np.zeros((len(LEADS), 2500), dtype=bool)
    status: dict[str, str] = {}
    for lead_index, lead in enumerate(LEADS):
        runs = contiguous_runs(mask500[:, lead_index])
        if not runs:
            raise ExtractionError(f"Lead {lead} has no observed waveform segment.")
        start, end = max(runs, key=lambda pair: pair[1] - pair[0])
        segment500 = canonical500[start:end, lead_index]
        segment250 = scipy.signal.resample_poly(segment500, 1, 2).astype(np.float32)
        if clean:
            try:
                segment250 = np.asarray(
                    nk.ecg_clean(segment250, sampling_rate=TRACE_FS), dtype=np.float32
                )
                status[lead] = "neurokit_ecg_clean"
            except Exception as exc:  # retain usable data while recording the fallback
                status[lead] = f"raw_resampled_fallback:{type(exc).__name__}"
        else:
            status[lead] = "raw_resampled"

        repetitions = int(math.ceil(2500 / segment250.size))
        tiled = np.tile(segment250, repetitions)[:2500]
        mean = float(np.mean(tiled))
        std = float(np.std(tiled))
        if not np.isfinite(std) or std < 1e-8:
            raise ExtractionError(f"Lead {lead} is constant or non-finite after reconstruction.")
        trace[lead_index] = (tiled - mean) / (std + 1e-6)

        start250 = int(round(start / 2.0))
        end250 = min(2500, start250 + segment250.size)
        valid_mask[lead_index, start250:end250] = True
    return trace, valid_mask, status


def valid_intervals(mask500: np.ndarray) -> dict[str, list[list[float]]]:
    result: dict[str, list[list[float]]] = {}
    for lead_index, lead in enumerate(LEADS):
        result[lead] = [
            [round(start / RECONSTRUCTION_FS, 3), round(end / RECONSTRUCTION_FS, 3)]
            for start, end in contiguous_runs(mask500[:, lead_index])
        ]
    return result


def rounded_for_json(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: rounded_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded_for_json(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any], private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rounded_for_json(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if private:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def record_paths(output_root: Path, private_root: Path, ecg_id: int) -> dict[str, Path]:
    bucket = f"{(ecg_id // 1000) * 1000:05d}"
    stem = f"{ecg_id:05d}"
    return {
        "bucket": Path(bucket),
        "stem": Path(stem),
        "record500": output_root / "records500" / bucket,
        "record100": output_root / "records100" / bucket,
        "mask": output_root / "validity_masks" / bucket / f"{stem}_mask.npz",
        "public_qc": output_root / "quality_control" / "records" / bucket / f"{stem}.json",
        "public_meta": output_root / "quality_control" / "metadata" / bucket / f"{stem}.json",
        "trace_record": private_root / "checkpoints" / "trace_records" / bucket / f"{stem}.npy",
        "trace_mask": private_root / "checkpoints" / "trace_masks" / bucket / f"{stem}.npy",
        "private_meta": private_root / "checkpoints" / "metadata" / bucket / f"{stem}.json",
    }


def public_checkpoint_complete(paths: dict[str, Path], ecg_id: int) -> bool:
    stem = f"{ecg_id:05d}"
    required = (
        paths["record500"] / f"{stem}_hr.hea",
        paths["record500"] / f"{stem}_hr.dat",
        paths["record100"] / f"{stem}_lr.hea",
        paths["record100"] / f"{stem}_lr.dat",
        paths["mask"],
        paths["public_qc"],
        paths["public_meta"],
        paths["trace_record"],
        paths["trace_mask"],
        paths["private_meta"],
    )
    return all(path.exists() for path in required)


def write_wfdb(
    directory: Path, record_name: str, fs: int, signal: np.ndarray
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    wfdb.wrsamp(
        record_name=record_name,
        fs=fs,
        units=["mV"] * len(LEADS),
        sig_name=list(LEADS),
        p_signal=signal,
        fmt=["16"] * len(LEADS),
        adc_gain=[1000.0] * len(LEADS),
        baseline=[0] * len(LEADS),
        write_dir=str(directory),
    )


def process_one(
    entry: SourceEntry,
    output_root_text: str,
    private_root_text: str,
    secret_hex: str,
    resume: bool,
    clean_trace: bool,
) -> tuple[int, str]:
    output_root = Path(output_root_text)
    private_root = Path(private_root_text)
    paths = record_paths(output_root, private_root, entry.ecg_id)
    if resume and public_checkpoint_complete(paths, entry.ecg_id):
        return entry.ecg_id, "resumed"

    secret = bytes.fromhex(secret_hex)
    source_path = Path(entry.source_path)
    with fitz.open(source_path) as doc:
        page = doc[0]
        canonical500, mask500, waveform_qc, _ = extract_waveforms(page, entry.template)
        canonical100, mask100 = canonical_100hz(canonical500, mask500)
        diagnosis = extract_diagnosis(page, entry.template)
        title = str((doc.metadata or {}).get("title") or "")
        age, sex, hospital_id, patient_source = extract_demographics_and_hospital_id(page, title)

    patient_key = hospital_id if hospital_id else f"record:{entry.source_sha256}"
    patient_id = "AZP" + keyed_token(secret, "patient", patient_key)
    scp_codes, mapping_status = map_scp_codes(diagnosis)
    trace, trace_mask, cleaning_status = trace_compatible(canonical500, mask500, clean=clean_trace)

    stem = f"{entry.ecg_id:05d}"
    write_wfdb(paths["record500"], f"{stem}_hr", RECONSTRUCTION_FS, canonical500)
    write_wfdb(paths["record100"], f"{stem}_lr", 100, canonical100)
    paths["mask"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths["mask"], mask500=mask500, mask100=mask100, leads=np.asarray(LEADS))
    paths["trace_record"].parent.mkdir(parents=True, exist_ok=True)
    paths["trace_mask"].parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["trace_record"], trace, allow_pickle=False)
    np.save(paths["trace_mask"], trace_mask, allow_pickle=False)

    public_meta = {
        "ecg_id": entry.ecg_id,
        "record_uid": entry.record_uid,
        "patient_id": patient_id,
        "patient_id_reliability": patient_source,
        "age": age,
        "age_is_90_plus": bool(age == 90),
        "sex": sex,
        "report": diagnosis,
        "diagnosis_text": diagnosis,
        "scp_codes": scp_codes,
        "mapping_status": mapping_status,
        "source_template": waveform_qc["template"],
        "layout_subtype": waveform_qc["layout_subtype"],
        "reconstruction_fs": RECONSTRUCTION_FS,
        "source_representation": "pdf_vector_reconstructed_lead_segments",
        "is_original_digital_waveform": False,
        "is_fully_synchronous_12lead_10s": False,
        "valid_intervals_seconds": valid_intervals(mask500),
        "recommended_split": "external_test",
        "strat_fold": 10,
        "filename_lr": f"records100/{paths['bucket']}/{stem}_lr",
        "filename_hr": f"records500/{paths['bucket']}/{stem}_hr",
    }
    public_qc = {
        "ecg_id": entry.ecg_id,
        **waveform_qc,
        "wfdb500_shape": [5000, 12],
        "wfdb100_shape": [1000, 12],
        "trace_shape": [12, 2500],
        "trace_cleaning": cleaning_status,
        "trace_mean_max_abs": float(np.max(np.abs(trace.mean(axis=1)))),
        "trace_std_max_abs_error": float(np.max(np.abs(trace.std(axis=1) - 1.0))),
        "canonical500_finite_observed": bool(np.isfinite(canonical500[mask500]).all()),
        "canonical500_nan_unobserved": bool(np.isnan(canonical500[~mask500]).all()),
    }
    private_meta = {
        "ecg_id": entry.ecg_id,
        "record_uid": entry.record_uid,
        "patient_id": patient_id,
        "source_relative_path": entry.relative_path,
        "source_sha256": entry.source_sha256,
        "hospital_patient_id": hospital_id,
        "patient_id_source": patient_source,
    }
    write_json(paths["public_meta"], public_meta)
    write_json(paths["public_qc"], public_qc)
    write_json(paths["private_meta"], private_meta, private=True)
    return entry.ecg_id, "processed"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ptb_metadata_row(meta: dict[str, Any]) -> dict[str, Any]:
    row = {
        "ecg_id": meta["ecg_id"],
        "patient_id": meta["patient_id"],
        "age": meta["age"],
        "sex": meta["sex"],
        "height": "",
        "weight": "",
        "nurse": "",
        "site": "",
        "device": meta["source_template"],
        "recording_date": "",
        "report": meta["report"],
        "scp_codes": repr(meta["scp_codes"]),
        "heart_axis": "",
        "infarction_stadium1": "",
        "infarction_stadium2": "",
        "validated_by": "",
        "second_opinion": "",
        "initial_autogenerated_report": "",
        "validated_by_human": False,
        "baseline_drift": "",
        "static_noise": "",
        "burst_noise": "",
        "electrodes_problems": "",
        "extra_beats": "",
        "pacemaker": "",
        "strat_fold": meta["strat_fold"],
        "filename_lr": meta["filename_lr"],
        "filename_hr": meta["filename_hr"],
        "record_uid": meta["record_uid"],
        "diagnosis_text": meta["diagnosis_text"],
        "mapping_status": meta["mapping_status"],
        "source_representation": meta["source_representation"],
        "source_template": meta["source_template"],
        "layout_subtype": meta["layout_subtype"],
        "reconstruction_fs": meta["reconstruction_fs"],
        "is_original_digital_waveform": meta["is_original_digital_waveform"],
        "is_fully_synchronous_12lead_10s": meta["is_fully_synchronous_12lead_10s"],
        "valid_intervals_seconds": json.dumps(meta["valid_intervals_seconds"], ensure_ascii=False),
        "recommended_split": meta["recommended_split"],
        "patient_id_reliability": meta["patient_id_reliability"],
        "age_is_90_plus": meta["age_is_90_plus"],
    }
    return row


def build_statement_table(metadata: pd.DataFrame, ptbxl_root: Path, output_root: Path) -> None:
    used: set[str] = set()
    for value in metadata["scp_codes"]:
        used.update(ast.literal_eval(value).keys())
    source = pd.read_csv(ptbxl_root / "scp_statements.csv", index_col=0)
    missing = sorted(used.difference(source.index))
    if missing:
        raise RuntimeError(f"Mapped SCP codes missing from PTB-XL statement table: {missing}")
    subset = source.loc[sorted(used)].copy() if used else source.iloc[0:0].copy()
    subset.index.name = "scp_code"
    subset.to_csv(output_root / "anzhen_statements.csv")
    subset.to_csv(output_root / "trace_compat" / "scp_statements.csv")


def build_evaluation_subsets(
    metadata: pd.DataFrame, ptbxl_root: Path, trace_dir: Path
) -> None:
    statements = pd.read_csv(ptbxl_root / "scp_statements.csv", index_col=0)
    diagnostic = statements[statements["diagnostic"] == 1.0]
    rows: list[dict[str, Any]] = []
    for row_index_value, row in metadata.reset_index(drop=True).iterrows():
        codes = ast.literal_eval(row["scp_codes"])
        superclasses = sorted(
            {
                str(diagnostic.loc[code, "diagnostic_class"])
                for code in codes
                if code in diagnostic.index
            }
        )
        report = "" if pd.isna(row["report"]) else str(row["report"]).strip()
        rows.append(
            {
                "row_index": row_index_value,
                "ecg_id": int(row["ecg_id"]),
                "record_uid": row["record_uid"],
                "has_trace_input": True,
                "has_diagnosis_text": bool(report),
                "has_any_mapped_scp_code": bool(codes),
                "has_five_superclass_target": bool(superclasses),
                "diagnostic_superclasses": repr(superclasses),
                "has_age": not pd.isna(row["age"]),
                "has_sex": not pd.isna(row["sex"]),
                "recommended_split": "external_test",
            }
        )
    pd.DataFrame(rows).to_csv(trace_dir / "evaluation_subsets.csv", index=False)


def assemble_outputs(
    entries: Sequence[SourceEntry], output_root: Path, private_root: Path, ptbxl_root: Path
) -> None:
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    trace_dir = output_root / "trace_compat"
    trace_dir.mkdir(parents=True, exist_ok=True)
    full_data = np.lib.format.open_memmap(
        trace_dir / "full_data.npy", mode="w+", dtype=np.float32, shape=(len(entries), 12, 2500)
    )
    full_mask = np.lib.format.open_memmap(
        trace_dir / "valid_mask.npy", mode="w+", dtype=bool, shape=(len(entries), 12, 2500)
    )

    row_index: list[dict[str, Any]] = []
    for row_index_value, entry in enumerate(entries):
        paths = record_paths(output_root, private_root, entry.ecg_id)
        if not public_checkpoint_complete(paths, entry.ecg_id):
            raise RuntimeError(f"Record {entry.ecg_id} is incomplete; cannot assemble release outputs.")
        public_meta = load_json(paths["public_meta"])
        private_meta = load_json(paths["private_meta"])
        public_qc = load_json(paths["public_qc"])
        public_rows.append(ptb_metadata_row(public_meta))
        private_rows.append(private_meta)
        qc_rows.append(
            {
                "ecg_id": entry.ecg_id,
                "template": public_qc["template"],
                "layout_subtype": public_qc["layout_subtype"],
                "standard_path_count": public_qc["standard_path_count"],
                "rhythm_path_count": public_qc["rhythm_path_count"],
                "observed_fraction": public_qc["observed_fraction"],
                "overlap_correlation": json.dumps(public_qc["overlap_correlation"]),
                "trace_mean_max_abs": public_qc["trace_mean_max_abs"],
                "trace_std_max_abs_error": public_qc["trace_std_max_abs_error"],
            }
        )
        full_data[row_index_value] = np.load(paths["trace_record"], allow_pickle=False)
        full_mask[row_index_value] = np.load(paths["trace_mask"], allow_pickle=False)
        row_index.append(
            {
                "row_index": row_index_value,
                "ecg_id": entry.ecg_id,
                "record_uid": entry.record_uid,
                "record_name": f"{entry.ecg_id:05d}",
            }
        )
    full_data.flush()
    full_mask.flush()

    metadata = pd.DataFrame(public_rows).sort_values("ecg_id")
    metadata.to_csv(output_root / "anzhen_database.csv", index=False)
    metadata.to_csv(trace_dir / "ptbxl_compat_database.csv", index=False)
    pd.DataFrame(row_index).to_csv(trace_dir / "row_index.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(output_root / "quality_control" / "record_qc.csv", index=False)

    private_mapping = private_root / "source_to_public_mapping.csv"
    pd.DataFrame(private_rows).sort_values("ecg_id").to_csv(private_mapping, index=False)
    os.chmod(private_mapping, stat.S_IRUSR | stat.S_IWUSR)
    build_statement_table(metadata, ptbxl_root, output_root)
    build_evaluation_subsets(metadata, ptbxl_root, trace_dir)


def wfdb_roundtrip_check(entries: Sequence[SourceEntry], output_root: Path) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    checked = 0
    for entry in entries:
        bucket = f"{(entry.ecg_id // 1000) * 1000:05d}"
        stem = f"{entry.ecg_id:05d}"
        for rate, suffix, expected_shape in ((500, "hr", (5000, 12)), (100, "lr", (1000, 12))):
            record = output_root / f"records{rate}" / bucket / f"{stem}_{suffix}"
            try:
                signal, fields = wfdb.rdsamp(str(record))
                if signal.shape != expected_shape or int(fields["fs"]) != rate:
                    failures.append(
                        {
                            "ecg_id": entry.ecg_id,
                            "rate": rate,
                            "shape": list(signal.shape),
                            "fs": fields["fs"],
                        }
                    )
            except Exception as exc:
                failures.append(
                    {"ecg_id": entry.ecg_id, "rate": rate, "error": f"{type(exc).__name__}: {exc}"}
                )
            checked += 1
    return {"checked_records": checked, "failure_count": len(failures), "failures": failures}


def write_records_file(entries: Sequence[SourceEntry], output_root: Path) -> None:
    lines: list[str] = []
    for entry in entries:
        bucket = f"{(entry.ecg_id // 1000) * 1000:05d}"
        stem = f"{entry.ecg_id:05d}"
        lines.append(f"records100/{bucket}/{stem}_lr")
    for entry in entries:
        bucket = f"{(entry.ecg_id // 1000) * 1000:05d}"
        stem = f"{entry.ecg_id:05d}"
        lines.append(f"records500/{bucket}/{stem}_hr")
    (output_root / "RECORDS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    private_root = args.private_root.resolve()
    ptbxl_root = args.ptbxl_root.resolve()
    if not input_root.is_dir():
        raise SystemExit(f"Input root does not exist: {input_root}")
    if not (ptbxl_root / "scp_statements.csv").is_file():
        raise SystemExit(f"PTB-XL statement table not found under: {ptbxl_root}")
    if output_root == input_root or private_root == input_root or output_root == private_root:
        raise SystemExit("Input, public output and private output roots must be different directories.")

    output_root.mkdir(parents=True, exist_ok=True)
    secret = ensure_private_root(private_root)
    selected = inventory_sources(input_root, private_root, args.pilot_per_template)
    if not selected:
        raise SystemExit("No supported resting ECG PDFs were found.")
    entries = build_source_entries(selected, input_root, secret)
    inventory_summary = {
        "dataset_version": PUBLIC_VERSION,
        "selected_records": len(entries),
        "pilot_per_template": args.pilot_per_template,
        "template_counts": dict(Counter(entry.template for entry in entries)),
        "original_pdfs_included_in_public_output": False,
    }
    write_json(output_root / "quality_control" / "build_inventory_summary.json", inventory_summary)

    print(f"[build] processing {len(entries)} resting ECG records with {args.workers} worker(s)", flush=True)
    failures: list[dict[str, Any]] = []
    if args.workers <= 1:
        for index, entry in enumerate(entries, start=1):
            try:
                _, status = process_one(
                    entry,
                    str(output_root),
                    str(private_root),
                    secret.hex(),
                    args.resume,
                    not args.skip_trace_cleaning,
                )
                print(f"[build] {index}/{len(entries)} ecg_id={entry.ecg_id} {status}", flush=True)
            except Exception as exc:
                failures.append(
                    {"ecg_id": entry.ecg_id, "error": f"{type(exc).__name__}: {exc}"}
                )
                print(f"[error] ecg_id={entry.ecg_id}: {type(exc).__name__}: {exc}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    process_one,
                    entry,
                    str(output_root),
                    str(private_root),
                    secret.hex(),
                    args.resume,
                    not args.skip_trace_cleaning,
                ): entry
                for entry in entries
            }
            completed = 0
            for future in as_completed(futures):
                entry = futures[future]
                completed += 1
                try:
                    _, status = future.result()
                    print(
                        f"[build] {completed}/{len(entries)} ecg_id={entry.ecg_id} {status}",
                        flush=True,
                    )
                except Exception as exc:
                    failures.append(
                        {"ecg_id": entry.ecg_id, "error": f"{type(exc).__name__}: {exc}"}
                    )
                    print(f"[error] ecg_id={entry.ecg_id}: {type(exc).__name__}: {exc}", flush=True)

    write_json(output_root / "quality_control" / "build_failures.json", {"failures": failures})
    if failures:
        print(f"[build] stopped before assembly: {len(failures)} record(s) failed", flush=True)
        return 2

    assemble_outputs(entries, output_root, private_root, ptbxl_root)
    write_records_file(entries, output_root)
    roundtrip = wfdb_roundtrip_check(entries, output_root)
    write_json(output_root / "quality_control" / "wfdb_roundtrip.json", roundtrip)
    if roundtrip["failure_count"]:
        print(f"[qc] WFDB round-trip failures: {roundtrip['failure_count']}", flush=True)
        return 3

    print("[build] dataset assembly and WFDB round-trip checks completed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
