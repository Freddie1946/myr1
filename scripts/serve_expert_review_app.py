#!/usr/bin/env python3
"""Serve the expert-review portal and persist strict six-event ratings."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


EVENTS = (
    "image_feature_analysis_present",
    "option_elimination_present",
    "medical_knowledge_support_present",
    "histological_definition_error",
    "logical_contradiction",
    "outdated_or_incorrect_pathology_criterion",
)
REVIEWER_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
CASE_RE = re.compile(r"^HRA-(?:00[1-9]|0[1-5][0-9]|060)$")
MAX_REQUEST_BYTES = 32 * 1024


def process_score(events: dict[str, bool]) -> float:
    missing = sum(not events[field] for field in EVENTS[:3])
    errors = sum(events[field] for field in EVENTS[3:])
    return (max(0.0, 1.0 - 0.4 * missing) + max(0.0, 1.0 - 0.4 * errors)) / 2.0


def validate_identity(reviewer_id: str, case_id: str) -> None:
    if not REVIEWER_RE.fullmatch(reviewer_id):
        raise ValueError("reviewer_id must contain only letters, numbers, period, underscore, or hyphen")
    if not CASE_RE.fullmatch(case_id):
        raise ValueError("case_id must be HRA-001 through HRA-060")


class RatingStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def path(self, reviewer_id: str, case_id: str) -> Path:
        validate_identity(reviewer_id, case_id)
        return self.root / reviewer_id / f"{case_id}.json"

    def load(self, reviewer_id: str, case_id: str) -> dict:
        path = self.path(reviewer_id, case_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, payload: dict) -> tuple[dict, int]:
        if not isinstance(payload, dict) or set(payload) != {"reviewer_id", "case_id", "events", "reviewer_notes"}:
            raise ValueError("request must contain reviewer_id, case_id, events, and reviewer_notes")
        reviewer_id = payload["reviewer_id"]
        case_id = payload["case_id"]
        if not isinstance(reviewer_id, str) or not isinstance(case_id, str):
            raise ValueError("reviewer_id and case_id must be strings")
        validate_identity(reviewer_id, case_id)
        events = payload["events"]
        if not isinstance(events, dict) or set(events) != set(EVENTS):
            raise ValueError("events must contain exactly the frozen six fields")
        if any(type(events[field]) is not bool for field in EVENTS):
            raise ValueError("all six event values must be exact booleans")
        notes = payload["reviewer_notes"]
        if not isinstance(notes, str) or len(notes) > 2000:
            raise ValueError("reviewer_notes must be a string of at most 2000 characters")
        record = {
            "schema_version": 1,
            "rating_contract": "pathvlm_stage3_process_events_v1_exact_six_booleans",
            "reviewer_id": reviewer_id,
            "case_id": case_id,
            "events": {field: events[field] for field in EVENTS},
            "process_score": process_score(events),
            "reviewer_notes": notes,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.lock:
            path = self.path(reviewer_id, case_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{case_id}.", suffix=".tmp", dir=path.parent)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(record, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, path)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
            self._write_csv(reviewer_id)
            completed = len(list((self.root / reviewer_id).glob("HRA-*.json")))
        return record, completed

    def _write_csv(self, reviewer_id: str) -> None:
        directory = self.root / reviewer_id
        records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("HRA-*.json"))]
        target = directory / "reward_six_event_ratings.csv"
        descriptor, temporary_name = tempfile.mkstemp(prefix=".ratings.", suffix=".tmp", dir=directory)
        fields = ["case_id", "reviewer_id", *EVENTS, "process_score", "reviewer_notes", "saved_at"]
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for record in records:
                    writer.writerow({
                        "case_id": record["case_id"],
                        "reviewer_id": reviewer_id,
                        **record["events"],
                        "process_score": f'{float(record["process_score"]):.6f}',
                        "reviewer_notes": record["reviewer_notes"],
                        "saved_at": record["saved_at"],
                    })
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, target)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)


def make_handler(directory: Path, store: RatingStore):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def json_response(self, status: HTTPStatus, value: dict) -> None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/reward-rating":
                return super().do_GET()
            query = parse_qs(parsed.query)
            reviewer_id = query.get("reviewer_id", [""])[0]
            case_id = query.get("case_id", [""])[0]
            try:
                self.json_response(HTTPStatus.OK, store.load(reviewer_id, case_id))
            except FileNotFoundError:
                self.json_response(HTTPStatus.NOT_FOUND, {"error": "rating not found"})
            except ValueError as error:
                self.json_response(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/reward-rating":
                self.json_response(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                record, completed = store.save(payload)
                self.json_response(HTTPStatus.OK, {
                    "status": "saved",
                    "case_id": record["case_id"],
                    "process_score": record["process_score"],
                    "saved_at": record["saved_at"],
                    "completed_cases": completed,
                })
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                self.json_response(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def log_message(self, format: str, *args) -> None:
            print(f"[{self.log_date_time_string()}] {format % args}", flush=True)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not (args.root / "index.html").is_file():
        raise FileNotFoundError(args.root / "index.html")
    store = RatingStore(args.output)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.root.resolve(), store))
    print(json.dumps({
        "status": "serving",
        "url": f"http://{args.host}:{args.port}/",
        "ratings_output": str(args.output.resolve()),
    }, ensure_ascii=False), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
