# Manuscript and reviewer-material handoff

This directory carries the paper, editorial decision, reviewer comments, and the revision records
needed by a new Codex session on the formal machine. The repository owner explicitly approved
placing these materials in the public Git repository on 2026-07-13.

## Mandatory reading order

1. `paper/PathVLM-R1_manuscript_extracted.txt`
2. `reviews/JBHI-06328-2025_decision_and_reviews_extracted.txt`
3. `reviews/reviewer_1_notes.md`
4. `revision/revision_todo_priorities.md`
5. `revision/revision_execution_log.md`
6. `revision/code_version_audit.md`
7. `environment/baseline_repos_and_envs.md`
8. `environment/reproducible_env_commands.md`

Then compare the reviewer requests with `../protocol/training_plan.md`, `../docs/LATEST.md`, and
the current run manifests. A historical note is not proof that an experiment is valid or complete.

## File map

- `paper/PathVLM-R1_manuscript.pdf`: authoritative submitted/revision manuscript PDF, 12 pages.
- `paper/PathVLM-R1_manuscript_extracted.txt`: searchable Poppler text extraction.
- `reviews/JBHI-06328-2025_decision_and_reviews.pdf`: authoritative editorial email and reviews.
- `reviews/JBHI-06328-2025_decision_and_reviews_extracted.txt`: searchable Poppler extraction.
- `reviews/reviewer_1_notes.md`: separately prepared Chinese Reviewer 1 notes.
- `revision/`: prior revision priorities, execution log, and code-version audit.
- `environment/`: prior environment and baseline-repository notes.
- `manifest.json`: SHA-256 provenance for every imported source/extracted artifact.

The PDFs are authoritative. Extracted text is a search convenience and may flatten columns,
figures, equations, or email layout. Consult the matching PDF before quoting or editing the paper.

## Scientific handling

- Reviewer requests do not override the frozen data/test protocol in `AGENTS.md`.
- Do not claim an experiment is complete merely because it appears in an old todo or execution log.
- Reconcile terminology with current code and manifests; document every material correction.
- Keep test evaluation sealed until prompts, checkpoints, and scoring are frozen.
