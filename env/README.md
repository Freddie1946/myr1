# Environment snapshots

Each formal run copies the following into its `env_snapshot/` directory:

- `pip_freeze.txt`
- `conda_explicit.txt`
- `nvidia_smi.txt`
- `torch_cuda_versions.txt`
- `llamafactory_git.txt`
- `git_status.txt`
- DeepSpeed configuration used by the run
- base model manifest
- data manifest

Do not update packages inside an environment after formal runs begin. Create a new named environment version instead.

