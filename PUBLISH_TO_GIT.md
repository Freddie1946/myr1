# Publish this handoff repository

The local repository contains no configured remote. After creating an empty private repository on the Git service accessible from both machines, run:

```bash
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin codex/formal-machine-handoff
```

On the formal machine:

```bash
git clone --branch codex/formal-machine-handoff <YOUR_REPOSITORY_URL> pathvlm-r1-revision
cd pathvlm-r1-revision
```

If you prefer `main`, merge the handoff branch through your normal review flow, then clone `main`.

Keep the repository private if the split data or paper revision materials are not intended for public release. Never commit model weights, images, credentials, API tokens, or machine-local `formal_machine.env`.

