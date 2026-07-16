#!/usr/bin/env python3
"""Audited two-tier retention for formal model checkpoints."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import threading
from datetime import datetime
from pathlib import Path


CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")
MODEL_AUXILIARY_FILES = {
    "README.md",
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "trainer_state.json",
    "training_args.bin",
    "video_preprocessor_config.json",
    "vocab.json",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    if not match:
        raise ValueError(f"not a checkpoint directory: {path}")
    return int(match.group(1))


def snapshot_source_files(checkpoint: Path) -> list[Path]:
    """Return the complete model-only file set after a checkpoint is fully written."""
    checkpoint = checkpoint.resolve()
    step = checkpoint_step(checkpoint)
    index_path = checkpoint / "model.safetensors.index.json"
    trainer_state_path = checkpoint / "trainer_state.json"
    if not index_path.is_file() or not trainer_state_path.is_file():
        raise FileNotFoundError(f"checkpoint is not complete: {checkpoint}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"model index has no weight map: {index_path}")
    state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    if state.get("global_step") != step:
        raise ValueError(
            f"trainer-state step mismatch for {checkpoint}: {state.get('global_step')} != {step}"
        )
    names = {"model.safetensors.index.json", *weight_map.values()}
    names.update(name for name in MODEL_AUXILIARY_FILES if (checkpoint / name).is_file())
    required_auxiliary = {
        "config.json",
        "merges.txt",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "trainer_state.json",
        "vocab.json",
    }
    missing_auxiliary = sorted(name for name in required_auxiliary if name not in names)
    if missing_auxiliary:
        raise FileNotFoundError(f"checkpoint lacks required model files: {missing_auxiliary}")
    sources = [checkpoint / name for name in sorted(names)]
    missing = [str(path) for path in sources if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError(f"checkpoint model files are missing or empty: {missing}")
    return sources


def _link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.EMLINK}:
            raise
        shutil.copy2(source, destination)
        return "copy"


def archive_checkpoint(
    checkpoint: Path,
    snapshot_root: Path,
    *,
    minimum_free_bytes: int = 0,
) -> dict:
    """Atomically preserve a model-only snapshot from a resumable checkpoint."""
    checkpoint = checkpoint.resolve()
    snapshot_root = snapshot_root.resolve()
    step = checkpoint_step(checkpoint)
    destination = snapshot_root / checkpoint.name
    if destination.exists():
        return validate_snapshot(destination)
    if minimum_free_bytes and shutil.disk_usage(snapshot_root.parent).free < minimum_free_bytes:
        raise RuntimeError(
            f"free disk is below retention reserve before archiving {checkpoint.name}"
        )
    sources = snapshot_source_files(checkpoint)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    temporary = snapshot_root / f".{checkpoint.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    if temporary.exists():
        raise FileExistsError(f"temporary snapshot already exists: {temporary}")
    temporary.mkdir()
    try:
        files = []
        for source in sources:
            target = temporary / source.name
            transfer = _link_or_copy(source, target)
            files.append({
                "name": source.name,
                "size_bytes": target.stat().st_size,
                "sha256": sha256(target),
                "transfer": transfer,
            })
        state = json.loads((temporary / "trainer_state.json").read_text(encoding="utf-8"))
        payload = {
            "schema_version": 1,
            "snapshot_id": checkpoint.name,
            "source_checkpoint": str(checkpoint),
            "created_at": now_iso(),
            "global_step": step,
            "epoch": state.get("epoch"),
            "model_only": True,
            "resumable": False,
            "files": files,
            "total_file_bytes": sum(item["size_bytes"] for item in files),
        }
        manifest = temporary / "snapshot_manifest.json"
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    # The files were hashed immediately before the atomic rename. A complete
    # hash revalidation is deliberately deferred to the terminal run gate so
    # training does not read every 16-GiB snapshot twice at each epoch.
    return payload


def validate_snapshot(snapshot: Path) -> dict:
    snapshot = snapshot.resolve()
    manifest_path = snapshot / "snapshot_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("model_only") is not True or payload.get("resumable") is not False:
        raise ValueError(f"invalid snapshot class: {manifest_path}")
    if payload.get("snapshot_id") != snapshot.name:
        raise ValueError(f"snapshot identity mismatch: {manifest_path}")
    if payload.get("global_step") != checkpoint_step(snapshot):
        raise ValueError(f"snapshot step mismatch: {manifest_path}")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"snapshot contains no files: {manifest_path}")
    recorded_names = {item.get("name") for item in files}
    if not {"model.safetensors.index.json", "config.json", "trainer_state.json"}.issubset(recorded_names):
        raise ValueError(f"snapshot lacks required files: {manifest_path}")
    for item in files:
        path = snapshot / item["name"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"]:
            raise FileNotFoundError(f"snapshot file size mismatch: {path}")
        if sha256(path) != item["sha256"]:
            raise ValueError(f"snapshot file hash mismatch: {path}")
    return payload


class TwoTierCheckpointArchiver:
    """Watch Trainer output and preserve every completed checkpoint as model-only."""

    def __init__(
        self,
        output_dir: Path,
        snapshot_root: Path,
        events_path: Path,
        *,
        minimum_free_bytes: int,
        poll_seconds: float = 5.0,
    ) -> None:
        self.output_dir = output_dir.resolve()
        self.snapshot_root = snapshot_root.resolve()
        self.events_path = events_path.resolve()
        self.minimum_free_bytes = minimum_free_bytes
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failure: BaseException | None = None
        self._archived: dict[int, dict] = {}
        self._observed_steps: set[int] = set()
        self._pending_errors: dict[int, str] = {}

    def scan_once(self) -> None:
        if not self.output_dir.is_dir():
            return
        checkpoints = sorted(
            (path for path in self.output_dir.glob("checkpoint-*") if path.is_dir()),
            key=checkpoint_step,
        )
        for checkpoint in checkpoints:
            step = checkpoint_step(checkpoint)
            self._observed_steps.add(step)
            if step in self._archived:
                continue
            if not (checkpoint / "trainer_state.json").is_file():
                continue
            try:
                payload = archive_checkpoint(
                    checkpoint, self.snapshot_root, minimum_free_bytes=self.minimum_free_bytes,
                )
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                # Trainer writes trainer_state.json before LLaMA-Factory finishes
                # saving processor/tokenizer files. Treat incomplete content as
                # pending and retry; the terminal missing-snapshot gate remains
                # fail-closed if the checkpoint rotates before becoming ready.
                self._pending_errors[step] = f"{type(exc).__name__}: {exc}"
                continue
            self._archived[step] = payload
            self._pending_errors.pop(step, None)
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "event": "model_snapshot_archived",
                    "timestamp": now_iso(),
                    "checkpoint": str(checkpoint),
                    "snapshot": str(self.snapshot_root / checkpoint.name),
                    "global_step": step,
                    "epoch": payload.get("epoch"),
                    "total_file_bytes": payload.get("total_file_bytes"),
                }) + "\n")

    def _run(self) -> None:
        try:
            while not self._stop.wait(self.poll_seconds):
                self.scan_once()
            self.scan_once()
        except BaseException as exc:
            self._failure = exc

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("checkpoint archiver was already started")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop_and_validate(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(600.0, self.poll_seconds * 3))
            if self._thread.is_alive():
                raise RuntimeError("checkpoint archiver did not stop")
        if self._failure is not None:
            raise RuntimeError(f"checkpoint archiver failed: {self._failure}") from self._failure
        self.scan_once()
        snapshots = []
        if self.snapshot_root.is_dir():
            for path in sorted(self.snapshot_root.glob("checkpoint-*"), key=checkpoint_step):
                snapshots.append({"path": str(path), **validate_snapshot(path)})
        return {
            "snapshot_root": str(self.snapshot_root),
            "count": len(snapshots),
            "snapshots": snapshots,
            "observed_checkpoint_steps": sorted(self._observed_steps),
            "missing_observed_snapshot_steps": sorted(
                self._observed_steps.difference(item["global_step"] for item in snapshots)
            ),
            "pending_errors": {
                str(step): message for step, message in sorted(self._pending_errors.items())
                if step not in {item["global_step"] for item in snapshots}
            },
        }
