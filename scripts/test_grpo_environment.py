#!/usr/bin/env python3

import accelerate
import datasets
import torch
import transformers
import trl
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig

from open_r1.trainer import Qwen2VLGRPOTrainer


print(
    "GRPO environment: PASS",
    {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "accelerate": accelerate.__version__,
        "datasets": datasets.__version__,
        "trainer": Qwen2VLGRPOTrainer.__name__,
    },
)

