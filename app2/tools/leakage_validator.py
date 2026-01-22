from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional


SHIFT_NEG_RE = re.compile(r"shift\(\s*-\d+")
SUSPICIOUS_COL_RE = re.compile(r"\b(future|lead|next|t\+\d+|fwd|ahead)\b", re.IGNORECASE)
SUSPICIOUS_SUFFIX_RE = re.compile(r"(_t\+\d+|_lead|_future|_next|_fwd)$", re.IGNORECASE)
TIME_COL_RE = re.compile(r"\b(time|date|dt|timestamp|ts|begin|end)\b", re.IGNORECASE)
MERGE_RE = re.compile(r"\b(merge|join)\s*\(")
MERGE_ASOF_RE = re.compile(r"\bmerge_asof\s*\(")
FILL_FORWARD_RE = re.compile(r"\b(bfill|backfill)\b", re.IGNORECASE)
ROLLING_CENTER_RE = re.compile(r"rolling\([^)]*center\s*=\s*True", re.IGNORECASE)
SHIFT_NEG_INDEXER_RE = re.compile(r"\.shift\(\s*-\d+\s*\)")

SEVERITY_MAP: Dict[str, str] = {
    "shift_negative": "high",
    "asof_forward": "high",
    "rolling_center": "medium",
    "time_merge_join": "medium",
    "backfill_usage": "medium",
    "suspicious_name": "low",
    "suspicious_column": "low",
}
SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}


@dataclass
class Finding:
    path: str
    kind: str
    severity: str
    detail: str


def _iter_files(paths: Iterable[str], extensions: List[str]) -> Iterable[str]:
    exts = {e.lower() for e in extensions}
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for name in files:
                    ext = os.path.splitext(name)[1].lower()
                    if ext in exts:
                        yield os.path.join(root, name)
        else:
            ext = os.path.splitext(p)[1].lower()
            if ext in exts and os.path.exists(p):
                yield p


def _scan_python(path: str) -> List[Finding]:
    findings: List[Finding] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return findings

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if SHIFT_NEG_RE.search(line) or SHIFT_NEG_INDEXER_RE.search(line):
            findings.append(
                Finding(path, "shift_negative", SEVERITY_MAP["shift_negative"], f"line {idx}: {line.strip()}")
            )
        if MERGE_RE.search(line) and TIME_COL_RE.search(line):
            findings.append(
                Finding(path, "time_merge_join", SEVERITY_MAP["time_merge_join"], f"line {idx}: {line.strip()}")
            )
        if MERGE_ASOF_RE.search(line) and re.search(r"direction\s*=\s*['\"](forward|nearest)['\"]", line):
            findings.append(
                Finding(path, "asof_forward", SEVERITY_MAP["asof_forward"], f"line {idx}: {line.strip()}")
            )
        if ROLLING_CENTER_RE.search(line):
            findings.append(
                Finding(path, "rolling_center", SEVERITY_MAP["rolling_center"], f"line {idx}: {line.strip()}")
            )
        if "rolling(" in line and "center=True" in line and not ROLLING_CENTER_RE.search(line):
            findings.append(
                Finding(path, "rolling_center", SEVERITY_MAP["rolling_center"], f"line {idx}: {line.strip()}")
            )
        if FILL_FORWARD_RE.search(line):
            findings.append(
                Finding(path, "backfill_usage", SEVERITY_MAP["backfill_usage"], f"line {idx}: {line.strip()}")
            )
        if stripped.startswith("#") or stripped.startswith("'''") or stripped.startswith('"""'):
            continue
        if '"' in line or "'" in line:
            if SUSPICIOUS_COL_RE.search(line):
                findings.append(
                    Finding(path, "suspicious_name", SEVERITY_MAP["suspicious_name"], f"line {idx}: {line.strip()}")
                )
    return findings


def _scan_csv_header(path: str) -> List[Finding]:
    findings: List[Finding] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
    except Exception:
        return findings

    if not header:
        return findings
    for col in header:
        if SUSPICIOUS_COL_RE.search(col) or SUSPICIOUS_SUFFIX_RE.search(col):
            findings.append(Finding(path, "suspicious_column", SEVERITY_MAP["suspicious_column"], f"column: {col}"))
    return findings


def _passes_allowlist(path: str, allowlist: List[str]) -> bool:
    if not allowlist:
        return True
    return any(re.search(pat, path) for pat in allowlist)


def _passes_min_severity(severity: str, min_severity: str) -> bool:
    return SEVERITY_RANK.get(severity, 0) >= SEVERITY_RANK.get(min_severity, 0)


def run(paths: List[str], extensions: List[str], allowlist: List[str], min_severity: str) -> Dict[str, object]:
    findings: List[Finding] = []
    for path in _iter_files(paths, extensions):
        if os.path.basename(path) == "leakage_validator.py":
            continue
        if not _passes_allowlist(path, allowlist):
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext == ".py":
            findings.extend(_scan_python(path))
        elif ext == ".csv":
            findings.extend(_scan_csv_header(path))

    findings = [f for f in findings if _passes_min_severity(f.severity, min_severity)]

    by_kind: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}
    for f in findings:
        by_kind[f.kind] = by_kind.get(f.kind, 0) + 1
        by_severity[f.severity] = by_severity.get(f.severity, 0) + 1

    return {
        "paths": paths,
        "extensions": extensions,
        "allowlist": allowlist,
        "min_severity": min_severity,
        "total_findings": len(findings),
        "findings_by_kind": by_kind,
        "findings_by_severity": by_severity,
        "findings": [asdict(f) for f in findings],
    }


def main(args):
    paths = list(args.paths)
    extensions = [e.strip() for e in args.extensions.split(",") if e.strip()]
    allowlist = [p.strip() for p in args.allowlist.split(",") if p.strip()] if args.allowlist else []
    min_severity = args.min_severity
    report = run(paths, extensions, allowlist, min_severity)
    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[leakage-check] saved report to {args.out}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return report
