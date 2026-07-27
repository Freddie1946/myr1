# A100 Stage 3 and evaluation machine: Codex start here

This is the authoritative entry point for the second formal machine introduced on
`2026-07-27`. The machine has eight A100 GPUs and the user workspace is:

```text
/home/dataset-assist-0/czy/wjy
```

The Git repository remains the scientific source of truth. Being on a different machine does not
reduce, replace, or silently cancel any experiment in the repository plan.

## 1. Two-machine responsibility boundary

### Existing formal machine

The existing machine is restricted to the next n=4000/continue-SFT training control after its
scientific definition, checkpoint schedule and storage policy are frozen. It must:

- perform the required training smoke/gates;
- train only the approved 4000-SFT control;
- preserve all predeclared epoch/checkpoint candidates, optimizer/training logs and manifests;
- not select a checkpoint with validation;
- not run Stage 3, baseline inference, OOD inference, validation or final test.

Whether “4000 SFT” means a fresh base-model run on all 4,000 SFT records or continued SFT from the
selected n=3000 parent on an additional 1,000 records is still a scientific decision. These are not
interchangeable controls. No launcher may be created or started until a timestamped protocol freezes
the choice.

### New eight-A100 machine

The new machine owns:

- all remaining Stage 2 work that is later retained in scope;
- validation and checkpoint selection for transferred 4000-SFT candidates;
- Stage 3 Process Reward design finalization, gates, training and the 0.3/0.4/0.5 sensitivity study;
- every manuscript baseline rerun that remains technically possible;
- the additional PLIP, CONCH, UNI and approved pathology/medical comparisons;
- PathMMU validation and the single locked final PathMMU test;
- Chest CT, ISIC2020, Retinal OCT-C8 and Diabetic Retinopathy reruns;
- PathVQA and approved additional PathMMU/OOD evaluations;
- reduced visual-dependence checks, statistics, raw-generation retention and both kinds of bad-case
  analysis;
- preparation of evidence for expert/human evaluation.

API models whose historical endpoint cannot be recovered remain recorded obligations. Provider
retirement may change an exact rerun into a documented contemporary-successor rerun, but must never
be hidden by renaming the successor as the historical model.

## 2. Personal Codex installation: do not share global Codex state

Use a dedicated `CODEX_HOME` and a user-owned install directory. This isolates configuration,
authentication, sessions, logs, skills, caches and standalone package metadata from global or other
users’ Codex state.

Run these commands on the A100 machine:

```bash
export WJY_WORK_ROOT=/home/dataset-assist-0/czy/wjy
export CODEX_HOME="$WJY_WORK_ROOT/.codex-wjy"
export CODEX_INSTALL_DIR="$WJY_WORK_ROOT/codex-bin"

mkdir -p "$WJY_WORK_ROOT" "$CODEX_HOME" "$CODEX_INSTALL_DIR"
chmod 700 "$CODEX_HOME" "$CODEX_INSTALL_DIR"

curl -fsSL https://chatgpt.com/codex/install.sh | sh

export PATH="$CODEX_INSTALL_DIR:$PATH"
command -v codex
codex --version
```

The `CODEX_HOME` directory must exist before invoking Codex. Do not copy another user’s
`auth.json`, sessions, global `AGENTS.md`, configuration or skills.

Create a minimal headless-server configuration:

```bash
umask 077
cat > "$CODEX_HOME/config.toml" <<'EOF'
approval_policy = "on-request"
sandbox_mode = "workspace-write"
cli_auth_credentials_store = "file"
web_search = "cached"

[sandbox_workspace_write]
network_access = true
EOF
chmod 600 "$CODEX_HOME/config.toml"
```

Do not use `danger-full-access`, `--yolo` or `approval_policy="never"` for this migration. The
launcher below grants only the user workspace as an additional writable directory.

Authenticate on a headless server with device authentication:

```bash
export WJY_WORK_ROOT=/home/dataset-assist-0/czy/wjy
export CODEX_HOME="$WJY_WORK_ROOT/.codex-wjy"
export CODEX_INSTALL_DIR="$WJY_WORK_ROOT/codex-bin"
export PATH="$CODEX_INSTALL_DIR:$PATH"

codex login --device-auth
codex login status
codex doctor
```

If device authentication is unavailable, run `codex login` and complete its browser flow. API-key
authentication is a separate billed mode and should only be used intentionally:

```bash
read -rsp 'OpenAI API key: ' OPENAI_API_KEY && echo
printf '%s' "$OPENAI_API_KEY" | codex login --with-api-key
unset OPENAI_API_KEY
```

Never place an OpenAI key/token in this repository, shell history, a prompt, a log, or the workspace
environment example.

Official Codex references used for this setup:

- `https://learn.chatgpt.com/docs/config-file/environment-variables`
- `https://learn.chatgpt.com/docs/auth`
- `https://learn.chatgpt.com/docs/config-file/config-basic`
- `https://learn.chatgpt.com/docs/agent-approvals-security`
- `https://learn.chatgpt.com/docs/developer-commands?surface=cli`

## 3. Clone the authoritative repository

The branch on GitHub must first contain this document and all prior local commits. Then run:

```bash
export WJY_WORK_ROOT=/home/dataset-assist-0/czy/wjy
git clone --branch codex/formal-machine-handoff --single-branch \
  https://github.com/Freddie1946/myr1.git \
  "$WJY_WORK_ROOT/myr1"
cd "$WJY_WORK_ROOT/myr1"
git status --short --branch
git log -1 --oneline
test -f A100_STAGE3_EVAL_CODEX_START.md
```

If the repository already exists, do not overwrite it. Inspect the worktree; only when clean:

```bash
cd /home/dataset-assist-0/czy/wjy/myr1
git fetch origin
git switch codex/formal-machine-handoff
git pull --ff-only origin codex/formal-machine-handoff
```

Use a dedicated execution branch to avoid two machines committing conflicting state:

```bash
git switch -c codex/a100-stage3-eval
```

The existing machine should use a separate `codex/sft4000` branch for the 4000-SFT attempt. Do not
force-push either branch. Integrate timestamped records explicitly after reviewing conflicts.

## 4. Create the A100 workspace environment

The repository contains a non-secret path template:

```bash
cd /home/dataset-assist-0/czy/wjy/myr1
cp formal_machine/a100_stage3_eval_workspace.env.example \
  /home/dataset-assist-0/czy/wjy/a100_stage3_eval_workspace.env
chmod 600 /home/dataset-assist-0/czy/wjy/a100_stage3_eval_workspace.env
source /home/dataset-assist-0/czy/wjy/a100_stage3_eval_workspace.env
mkdir -p "$PATHVLM_INSTALL_ROOT" "$PATHVLM_EVAL_ROOT" "$HF_HOME" \
  "$CONDA_PKGS_DIRS" "$TMPDIR"
```

This file contains paths only. Hugging Face, API and judge credentials must not be written into it.

Authenticate Hugging Face in the isolated cache after the user has accepted all required licenses:

```bash
source /home/dataset-assist-0/czy/wjy/a100_stage3_eval_workspace.env
hf auth login
hf auth whoami
```

If `hf` is not available, the new Codex may install `huggingface_hub[cli]` into a small
user/workspace-owned bootstrap environment. It must not install packages into another user’s
environment.

## 5. Start Codex and let it prepare the machine

The checked-in launcher always supplies the correct repository root, personal `CODEX_HOME`, sandbox
and writable workspace:

```bash
cd /home/dataset-assist-0/czy/wjy/myr1
bash scripts/launch_personal_codex_a100.sh
```

To start with the complete bootstrap prompt automatically:

```bash
cd /home/dataset-assist-0/czy/wjy/myr1
bash scripts/launch_personal_codex_a100.sh \
  "$(cat protocol/a100_codex_bootstrap_prompt_20260727.txt)"
```

Codex must read `AGENTS.md` and its mandatory reading order before acting. The bootstrap prompt
authorizes machine audit, pinned data/model downloads, isolated environment creation and integrity
checks after disk/access gates pass. It does not authorize training, validation inference, test
inference, Stage 3 calls, API spending, process termination or cleanup.

Approve only scoped commands belonging to the preparation plan. If a command proposes deleting
files, stopping processes, using occupied GPUs, opening test outputs, storing credentials in Git or
changing the frozen split, reject it.

## 6. Formal bootstrap paths on A100

After the new Codex audits the driver, CUDA compatibility, Conda and disk, it creates the ignored
`formal_machine.env` from `formal_machine/formal_machine.env.example` with at least:

```text
INSTALL_ROOT=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100
BUNDLE_ROOT=/home/dataset-assist-0/czy/wjy/myr1
SPLIT_ROOT=/home/dataset-assist-0/czy/wjy/myr1/data/pathmmu_image_disjoint_v2
IMAGE_ROOT=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/raw_data/pathmmu/images
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC_PER_NODE=8
```

`CONDA_EXE` must be discovered on the new machine. PyTorch 2.6.0/cu124 remains the audited default
only if the installed NVIDIA driver supports it; otherwise stop and document the incompatibility
instead of silently changing the formal environment.

Run:

```bash
source /home/dataset-assist-0/czy/wjy/a100_stage3_eval_workspace.env
cd "$PATHVLM_REPO_ROOT"
bash setup_formal_machine.sh formal_machine.env \
  2>&1 | tee "$PATHVLM_WORKSPACE_ROOT/logs/a100_formal_bootstrap_$(date +%Y%m%d_%H%M%S).log"
```

The preflight report must pass before any later smoke. Preparation also follows Section 5 of
`FORMAL_MACHINE_CODEX_GUIDE.md` for external datasets and baseline environments.

## 7. Transfer the 4000-SFT outputs without Git

Model weights and raw outputs never enter Git. The current machine must create a transfer manifest
containing:

- frozen 4000-SFT protocol and exact source-data hashes;
- base/parent identity and training seed;
- every retained epoch/checkpoint candidate and exact byte size/SHA-256;
- optimizer/training state retained for recovery;
- tokenizer, processor and `chat_template.json`;
- launcher/code/environment hashes and complete training log;
- confirmation that no validation/test result was used on the source machine.

Transfer through approved shared storage, `rsync`/SSH or an object store. On A100, hash every file
before loading it. Do not infer checkpoint identity from a directory name. A100 performs validation
selection only after the transfer manifest passes.

## 8. Experiment order on A100

Preparation and experiment execution are separate approvals:

1. bootstrap formal data/base environments and pass preflight;
2. reproduce small CPU/import and one-step A100 hardware gates;
3. prepare external datasets and all baseline-specific isolated environments;
4. receive and verify every 4000-SFT candidate;
5. freeze the common validation prompt, preprocessing, decoding, parser and raw-output schema;
6. validate 4000-SFT candidates and other selectable checkpoints;
7. freeze Stage 3 parent, process-event JSON schema, external scoring code, judge model/version,
   penalties, fallbacks and 0.3/0.4/0.5 sensitivity protocol;
8. run Stage 3 smoke, reward-variance/parser/tensor gates, then separately approved formal runs;
9. run all feasible original and added baselines with raw predictions;
10. run the four cross-modal/OOD datasets, PathVQA and approved additional pathology evaluations;
11. complete paired statistics, visual-dependence checks and bad-case strata;
12. lock every model/prompt/parser/threshold, then run PathMMU final test once;
13. prepare human/expert-evaluation packets without contaminating model selection.

All failures and unavailable historical APIs remain visible in the final evidence. Test is never
used to choose a prompt, seed, checkpoint, parser, reward, threshold or baseline adapter.

## 9. Stage 3 inherited draft, not yet a frozen protocol

The agreed direction keeps the original two-part judge rubric rather than replacing it with a very
different prompt:

- judge image-grounded thinking-chain integrity separately from final-answer correctness;
- check image-feature analysis, option-elimination reasoning and medical-knowledge support;
- check histological-definition errors, logical contradictions and outdated/incorrect criteria;
- require structured JSON-only events from the judge;
- compute numerical penalties and the final reward deterministically outside the judge;
- do not require an arbitrary fixed count of eliminated options.

The exact JSON schema, event counting/deduplication, penalty aggregation, judge model/version,
timeouts/retries/fallbacks, leakage controls and sensitivity analysis are still pending a new frozen
protocol. The A100 Codex must not invent them or claim historical recovery.
