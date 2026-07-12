#!/usr/bin/env python3
"""Download and extract exactly the PathMMU images used by the frozen split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset-id", default="jamessyx/PathMMU")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def required_image_names(split_root: Path) -> set[str]:
    split_dir = split_root / "splits"
    paths = sorted(split_dir.glob("*_with_cot.json"))
    if not paths:
        raise SystemExit(f"no *_with_cot.json files found under {split_dir}")
    names: set[str] = set()
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            name = Path(row["image"]).name
            if not name or name in {".", ".."}:
                raise SystemExit(f"invalid image path in {path}: {row.get('image')!r}")
            names.add(name)
    return names


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image(path: Path) -> None:
    from PIL import Image

    with Image.open(path) as image:
        image.verify()


def download_archive(args: argparse.Namespace) -> tuple[Path, str | None]:
    if args.archive:
        archive = args.archive.expanduser().resolve()
        if not archive.is_file():
            raise SystemExit(f"PathMMU archive does not exist: {archive}")
        return archive, None

    try:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required to download PathMMU") from exc

    token = os.environ.get(args.token_env) or None
    try:
        info = HfApi(token=token).dataset_info(args.dataset_id, revision=args.revision)
        archive = hf_hub_download(
            repo_id=args.dataset_id,
            repo_type="dataset",
            filename="images.zip",
            revision=args.revision,
            token=token,
            cache_dir=args.cache_dir,
        )
    except GatedRepoError as exc:
        raise SystemExit(
            "PathMMU access is gated. Complete the dataset application, accept the "
            "Hugging Face terms, then run `hf auth login` or export HF_TOKEN before retrying."
        ) from exc
    except HfHubHTTPError as exc:
        raise SystemExit(f"PathMMU download failed: {exc}") from exc
    return Path(archive).resolve(), info.sha


def extract_required(
    archive: Path, image_root: Path, required: set[str], overwrite: bool
) -> tuple[int, int, list[str]]:
    image_root.mkdir(parents=True, exist_ok=True)
    existing = 0
    needed = set(required)
    if not overwrite:
        for name in sorted(required):
            target = image_root / name
            if target.is_file():
                try:
                    validate_image(target)
                except Exception as exc:
                    raise SystemExit(f"existing image is corrupt: {target}: {exc}") from exc
                existing += 1
                needed.remove(name)

    extracted = 0
    seen: dict[str, str] = {}
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if member.is_dir():
                continue
            name = Path(member.filename).name
            if name not in needed:
                continue
            if name in seen:
                raise SystemExit(
                    f"duplicate basename {name!r} in archive: {seen[name]!r}, {member.filename!r}"
                )
            seen[name] = member.filename
            target = image_root / name
            partial = image_root / f".{name}.part"
            with bundle.open(member) as source, partial.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            try:
                validate_image(partial)
                partial.replace(target)
            except Exception as exc:
                partial.unlink(missing_ok=True)
                raise SystemExit(f"invalid image extracted for {name}: {exc}") from exc
            extracted += 1

    missing = sorted(name for name in required if not (image_root / name).is_file())
    return existing, extracted, missing


def main() -> int:
    args = parse_args()
    required = required_image_names(args.split_root)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    archive, resolved_revision = download_archive(args)
    existing, extracted, missing = extract_required(
        archive, args.image_root, required, args.overwrite
    )
    report = {
        "dataset_id": args.dataset_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "archive": str(archive),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "required_unique_images": len(required),
        "existing_valid_images": existing,
        "newly_extracted_images": extracted,
        "missing_count": len(missing),
        "missing_images": missing,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "missing_images"}, indent=2))
    if missing:
        print(
            f"ERROR: official images.zip is missing {len(missing)} required images; "
            f"see {args.report}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
