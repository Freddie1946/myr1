# Legacy Codex snapshot repository — do not restore

The private Hugging Face dataset `Freddie1946/PathVLM-R1-Codex-Private-Snapshots` is deprecated.
Its historical archives excluded `auth.json` but retained session/shell/log state containing exact
copies of two API credentials. Do not download, restore, share, or promote any snapshot from that
repository.

Use only the sanitized replacement:

```text
dataset: Freddie1946/PathVLM-R1-Codex-Private-Snapshots-Clean-20260815
revision: 4c9e0778895d8a06fda45f391bbe64e7699fd634
prefix: snapshots/20260815T134000Z
archive sha256: 6e87ab9182a705e31e35fc11e30606f2d5d7f88e3ccc16d5021219b7b345c4ab
```

The clean snapshot excludes `auth.json`, the logs database and shell snapshots, redacts
token-shaped strings from copied text/session files, and passes both secret scanning and JSONL
parsing. The affected OpenRouter and AIGCBest credentials must be rotated. Deleting the deprecated
repository is a separate destructive action requiring user approval.
