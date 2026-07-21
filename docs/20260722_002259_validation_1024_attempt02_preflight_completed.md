# Corrected 1024-token validation Attempt02 preflight completed

Timestamp: `2026-07-22T00:22:59+08:00`

The read-only Attempt02 preflight passed at repository commit `9a8cc52`. It verified the frozen
code-hash manifest, validation data identity and count, chat template, all 26 candidate checkpoint
identities, and their parent run manifests. The frozen generation limit is 1,024 new tokens and the
expected output count is 10,010 raw validation predictions. Test was not accessed.

Per explicit user instruction, the automated GPU inventory/process check was not executed. The
preflight records physical GPUs 0--7 as manually confirmed by the user. No model was loaded onto a
GPU and no training or inference was started during this gate.

The next authorized action is the from-scratch Attempt02 validation run. A 1,024-token timeout is
preserved and scored under unchanged parser v2, reported per model, and does not stop the queue.
All other integrity and execution failures remain fail-closed.
