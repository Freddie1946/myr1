# PathVLM-R1 revision handoff

This Git repository is a machine-independent handoff for configuring the formal training machine and continuing the PathVLM-R1 revision experiments without access to the original Codex chat memory.

Start with [CODEX_START_HERE.md](CODEX_START_HERE.md), then follow [AGENTS.md](AGENTS.md).

Human setup command:

```bash
cp formal_machine/formal_machine.env.example formal_machine.env
# Edit formal_machine.env.
bash setup_formal_machine.sh formal_machine.env
```

Model weights, images, environments, caches, and checkpoints are intentionally excluded from Git.
Online bootstrap downloads the exact model and the gated official PathMMU `images.zip` after the
user has obtained dataset access and authenticated with Hugging Face.
