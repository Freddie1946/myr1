# SFT4000 fixed checkpoint uploaded to private Hugging Face repository

Completed: `2026-07-30T21:42:44+08:00`

The fixed SFT4000 control epoch-2/step-250 model-only snapshot was uploaded to:

`Freddie1946/PathVLM-R1-SFT-n4000-control-seed42-epoch2`

Fixed revision:

`31ecd18b9dc9de5c4118efde586bb625e4a6da06`

The repository is private. It contains the full model-only Transformers snapshot, the original
snapshot manifest and a corrected model card describing the selected SFT3000 parent, exact RL1000
continued-SFT role and diagnostic evaluation limitations.

## Local source

`/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/stage2_control_sft4000/n1000_seed0042/sft4000_control_rl1000_seed0042_epoch02_20260729_212746/epoch_snapshots/checkpoint-250`

- model-only: true;
- resumable optimizer state: false;
- global step: 250;
- epoch: 2;
- snapshot-manifest SHA-256:
  `d966b80a86a74495f35e0a2f34e2c66e1347a674f6db8fa96ac922afa8dc5e1c`;
- model-index SHA-256:
  `3067e9b0f35596ff3426a0d0ec8c982a51fa1e110c4fc30dcf3be9ea37409df6`.

## Remote verification

Hugging Face returned:

- final revision: `31ecd18b9dc9de5c4118efde586bb625e4a6da06`;
- private: true;
- repository files: 20;
- used storage: 16,595,844,048 bytes;
- `README.md`: present;
- `snapshot_manifest.json`: present.

Every file declared by the local snapshot manifest is present remotely with the same size. The
remote LFS SHA-256 for all four safetensor shards exactly matches the local manifest:

- shard 1:
  `761cc264f1c44b833daf62193766d469b9afc2d8745a70a6a7409654f3cb0fe3`;
- shard 2:
  `58f44da7a633b594b02dd3dc499db3ae591e40dffc8266e1722b0c3b301895f6`;
- shard 3:
  `b0c099c6ba00ac79c696bd93115d1d064720b893d5939dcd30cb536cb69de184`;
- shard 4:
  `3e49f6b6e0f6ec9fa13d5c2fdc195a0742c655baab5d18454a196caa44bbd840`.

This verification checks the remote committed revision rather than treating a successful CLI exit
as sufficient evidence.
