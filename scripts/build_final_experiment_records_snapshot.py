#!/usr/bin/env python3
"""Build a final, secret-free snapshot of experiment records and evaluation outputs.

The snapshot preserves formal mainline/ablation training records, data contracts, evaluation
outputs, and the latest human-review/explainability material. Smoke/throughput-gate artefacts,
model weights, optimizer states, regenerable caches, third-party datasets, and credentials are
excluded. Paper validity is determined by the final audit and result lineage, not merely by a
file's presence in this archive.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


WORK = Path("/home/dataset-assist-0/czy/wjy")
REPO = WORK / "myr1"

META_EXTENSIONS = {
    ".json", ".jsonl", ".md", ".txt", ".log", ".csv", ".tsv", ".yaml", ".yml",
    ".toml", ".out", ".err", ".sha256", ".done", ".metadata", ".tag", ".html",
    ".htm", ".py", ".sh",
}
VISUAL_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DATA_CONTRACT_EXTENSIONS = {
    ".json", ".jsonl", ".md", ".txt", ".csv", ".tsv", ".yaml", ".yml", ".sha256",
}

ROOTS = {
    "evaluation_runs": (WORK / "pathvlm_revision_eval_a100/runs", META_EXTENSIONS | VISUAL_EXTENSIONS),
    "evaluation_reports": (WORK / "pathvlm_revision_eval_a100/reports", META_EXTENSIONS | VISUAL_EXTENSIONS),
    "human_review": (WORK / "pathvlm_revision_eval_a100/human_review", META_EXTENSIONS | VISUAL_EXTENSIONS),
    "training_runs": (WORK / "pathvlm_r1_v1_a100/runs", META_EXTENSIONS | {".png"}),
    "training_reports": (WORK / "pathvlm_r1_v1_a100/reports", META_EXTENSIONS | VISUAL_EXTENSIONS),
    "training_data_contracts": (WORK / "pathvlm_r1_v1_a100/data", DATA_CONTRACT_EXTENSIONS),
    "evaluation_data_contracts": (WORK / "pathvlm_revision_eval_a100/datasets", DATA_CONTRACT_EXTENSIONS),
    "generated_training_configs": (WORK / "pathvlm_r1_v1_a100/generated_configs", META_EXTENSIONS),
}

EXCLUDED_PARTS = {".cache", "__pycache__", ".git", ".secrets"}
EXCLUDED_EXACT_FILENAMES = {"auth.json", "credentials.json"}
TEXT_EXTENSIONS = META_EXTENSIONS | DATA_CONTRACT_EXTENSIONS

SECRET_PATTERNS = {
    "huggingface_token": re.compile(rb"hf_[A-Za-z0-9]{24,}"),
    "openai_style_key": re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    "bearer_token": re.compile(rb"(?i)authorization\s*[:=]\s*bearer\s+[A-Za-z0-9._~+/-]{16,}"),
    "assigned_api_key": re.compile(
        rb"(?i)(?:api[_-]?key|token)\s*[\"']?\s*[:=]\s*[\"'][A-Za-z0-9._~+/-]{16,}[\"']"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_selected() -> Iterable[tuple[str, Path, str]]:
    seen: set[Path] = set()
    for category, (root, extensions) in ROOTS.items():
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink() or path in seen:
                continue
            relative = path.relative_to(root)
            if any(part in EXCLUDED_PARTS for part in relative.parts):
                continue
            # Smoke and throughput-gate artefacts proved engineering viability but are not
            # scientific results or required migration assets. Match components rather than
            # substrings over the full path so ordinary filenames remain unaffected.
            if any("smoke" in part.lower() for part in relative.parts):
                continue
            if path.name in EXCLUDED_EXACT_FILENAMES or path.suffix.lower() not in extensions:
                continue
            # Browser-function test records are not formal expert ratings and have their own warning
            # in the human-review protocol. Exclude them from the final scientific snapshot.
            if category == "human_review" and "expert_submissions_20260815" in relative.parts:
                continue
            seen.add(path)
            yield category, path, f"records/{category}/{relative.as_posix()}"


def existing_backup_hashes() -> set[str]:
    values: set[str] = set()
    manifests = list((WORK / "backup_archives").glob("**/ARCHIVE_MANIFEST.json"))
    manifests += list((WORK / "pathvlm_r1_v1_a100/reports").glob("**/*manifest*.json"))
    manifests += list((REPO / "protocol").glob("*backup*.json"))

    def walk(value: object) -> None:
        if isinstance(value, dict):
            candidate = value.get("sha256")
            if isinstance(candidate, str) and re.fullmatch(r"[0-9a-f]{64}", candidate):
                values.add(candidate)
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    for manifest in manifests:
        try:
            walk(json.loads(manifest.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
    return values


def secret_scan(path: Path) -> list[str]:
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return []
    matches: set[str] = set()
    overlap = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            data = overlap + chunk
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(data):
                    matches.add(label)
            overlap = data[-512:]
    return sorted(matches)


def sanitized_json_bytes(path: Path) -> tuple[bytes, list[str]]:
    """Redact token-shaped strings without changing the source file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels: set[str] = set()

    def redact(value: object) -> object:
        if isinstance(value, dict):
            return {key: redact(nested) for key, nested in value.items()}
        if isinstance(value, list):
            return [redact(nested) for nested in value]
        if not isinstance(value, str):
            return value
        result = value
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(result.encode("utf-8")):
                labels.add(label)
                result = pattern.sub(f"[REDACTED:{label}]".encode(), result.encode("utf-8")).decode("utf-8")
        return result

    clean = (json.dumps(redact(payload), ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    for label, pattern in SECRET_PATTERNS.items():
        if pattern.search(clean):
            raise RuntimeError(f"redaction failed for {path}: {label}")
    return clean, sorted(labels)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    previous_hashes = existing_backup_hashes()
    entries: list[dict] = []
    secret_findings: list[dict] = []
    sanitized_payloads: dict[str, bytes] = {}
    category_summary: dict[str, dict[str, int]] = {}
    for category, path, archive_path in iter_selected():
        source_digest = sha256(path)
        findings = secret_scan(path)
        if findings:
            if path.suffix.lower() != ".json":
                secret_findings.append({"source": str(path), "patterns": findings, "action": "unresolved"})
                continue
            clean, redacted = sanitized_json_bytes(path)
            digest = hashlib.sha256(clean).hexdigest()
            archive_bytes = len(clean)
            sanitized_payloads[archive_path] = clean
            secret_findings.append({
                "source": str(path), "patterns": findings, "action": "redacted_in_archive",
                "source_sha256": source_digest, "archive_sha256": digest,
            })
        else:
            digest = source_digest
            archive_bytes = path.stat().st_size
        entry = {
            "category": category,
            "source": str(path),
            "archive_path": archive_path,
            "bytes": archive_bytes,
            "sha256": digest,
            "previously_backed_up_by_content_sha256": digest in previous_hashes,
        }
        if findings:
            entry.update({
                "sanitized": True,
                "source_bytes": path.stat().st_size,
                "source_sha256": source_digest,
                "redacted_patterns": redacted,
            })
        entries.append(entry)
        summary = category_summary.setdefault(category, {"files": 0, "bytes": 0, "new_files": 0, "new_bytes": 0})
        summary["files"] += 1
        summary["bytes"] += entry["bytes"]
        if not entry["previously_backed_up_by_content_sha256"]:
            summary["new_files"] += 1
            summary["new_bytes"] += entry["bytes"]

    now = datetime.now(timezone.utc).astimezone().isoformat()
    manifest = {
        "schema_version": 1,
        "created_at": now,
        "git_head": git_head(),
        "purpose": "final secret-free snapshot of experiment records, evaluation JSON/JSONL, and visual analysis outputs",
        "selection_policy": (
            "All current record/output files matching the documented extensions under evaluation runs/reports, "
            "canonical human-review material, formal mainline/ablation training runs and reports, derived training "
            "data contracts, evaluation data contracts, and generated configs. Smoke/throughput-gate artefacts are "
            "excluded; paper-valid results are defined by final_revision_assets_audit and result_lineage."
        ),
        "exclusions": [
            "model weights and adapters", "optimizer/scheduler/RNG/DeepSpeed states", "third-party dataset images/parquet",
            "download caches", "credentials and auth files", "browser-function test expert submissions",
            "smoke tests and throughput-gate artefacts",
        ],
        "contains_model_weights": False,
        "contains_credentials": False,
        "file_count": len(entries),
        "total_bytes": sum(item["bytes"] for item in entries),
        "previously_backed_up_file_count": sum(item["previously_backed_up_by_content_sha256"] for item in entries),
        "previously_backed_up_bytes": sum(
            item["bytes"] for item in entries if item["previously_backed_up_by_content_sha256"]
        ),
        "new_or_changed_file_count": sum(not item["previously_backed_up_by_content_sha256"] for item in entries),
        "new_or_changed_bytes": sum(
            item["bytes"] for item in entries if not item["previously_backed_up_by_content_sha256"]
        ),
        "category_summary": category_summary,
        "files": entries,
    }
    manifest_path = output / "FINAL_EXPERIMENT_RECORDS_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {
        "schema_version": 1,
        "created_at": now,
        "status": (
            "failed_secret_scan" if any(item["action"] == "unresolved" for item in secret_findings)
            else "inventory_verified_with_redactions" if secret_findings
            else "inventory_verified"
        ),
        "git_head": manifest["git_head"],
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "previously_backed_up_file_count": manifest["previously_backed_up_file_count"],
        "previously_backed_up_bytes": manifest["previously_backed_up_bytes"],
        "new_or_changed_file_count": manifest["new_or_changed_file_count"],
        "new_or_changed_bytes": manifest["new_or_changed_bytes"],
        "secret_scan_findings": secret_findings,
        "category_summary": category_summary,
    }
    audit_path = output / "FINAL_EXPERIMENT_RECORDS_AUDIT.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if audit["status"] == "failed_secret_scan":
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        raise SystemExit("secret scan failed; no archive created")
    if args.inventory_only:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = output / f"pathvlm_final_experiment_records_{stamp}.tar.gz"
    with tarfile.open(archive_path, "w:gz", compresslevel=3) as archive:
        for item in entries:
            clean = sanitized_payloads.get(item["archive_path"])
            if clean is None:
                archive.add(item["source"], arcname=item["archive_path"], recursive=False)
            else:
                info = tarfile.TarInfo(item["archive_path"])
                info.size = len(clean)
                info.mtime = int(Path(item["source"]).stat().st_mtime)
                info.mode = 0o600
                archive.addfile(info, io.BytesIO(clean))
        archive.add(manifest_path, arcname=manifest_path.name, recursive=False)
        archive.add(audit_path, arcname=audit_path.name, recursive=False)
    verification = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "archive": archive_path.name,
        "bytes": archive_path.stat().st_size,
        "sha256": sha256(archive_path),
        "manifest_sha256": sha256(manifest_path),
        "audit_sha256": sha256(audit_path),
        "file_count": manifest["file_count"],
        "total_uncompressed_bytes": manifest["total_bytes"],
    }
    (output / "FINAL_EXPERIMENT_RECORDS_VERIFICATION.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"audit": audit, "verification": verification}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
