import json
import sqlite3
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("create_codex_online_snapshot.py")


def test_snapshot_excludes_sensitive_state_and_redacts_session(tmp_path: Path) -> None:
    source = tmp_path / "codex-home"
    source.mkdir()
    (source / "auth.json").write_text('{"token":"must-not-copy"}\n', encoding="utf-8")
    (source / "config.toml").write_text("model = 'test'\n", encoding="utf-8")
    (source / "history.jsonl").write_text('{"text":"safe"}\n', encoding="utf-8")
    sessions = source / "sessions"
    sessions.mkdir()
    fake_key = "sk-" + "A" * 32
    (sessions / "session.jsonl").write_text(
        json.dumps({"message": f"credential {fake_key}"}) + "\n", encoding="utf-8"
    )
    shell = source / "shell_snapshots"
    shell.mkdir()
    (shell / "state.sh").write_text(f"export API_KEY='{fake_key}'\n", encoding="utf-8")

    for name in ("state_5.sqlite", "goals_1.sqlite", "memories_1.sqlite", "logs_2.sqlite"):
        with sqlite3.connect(source / name) as database:
            database.execute("create table audit(value text)")
            database.execute("insert into audit values (?)", (fake_key if name == "logs_2.sqlite" else "safe",))

    output = tmp_path / "snapshot"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--codex-home", str(source), "--output-dir", str(output)],
        check=True,
    )
    root = output / "codex-home-snapshot"
    assert not (root / "auth.json").exists()
    assert not (root / "logs_2.sqlite").exists()
    assert not (root / "shell_snapshots").exists()
    assert (root / "state_5.sqlite").exists()
    session_payload = (root / "sessions/session.jsonl").read_text(encoding="utf-8")
    assert fake_key not in session_payload
    assert "[REDACTED:openai_style_key]" in session_payload
    json.loads(session_payload)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["redactions"]
    assert "logs_2.sqlite" in manifest["excluded_additional"]
