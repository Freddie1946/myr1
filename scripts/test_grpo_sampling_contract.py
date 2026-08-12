import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from transformers import GenerationConfig

from grpo_pathmmu import (
    EpochSnapshotCallback,
    ExplicitSaveStepsCallback,
    ReadOnlySchedulerProxy,
    RelativeLinearScheduler,
    freeze_generation_contract,
    install_scheduler_restart_on_resume,
)


class SamplingContractTests(unittest.TestCase):
    def test_hardcoded_vendor_defaults_are_overridden(self):
        trainer = SimpleNamespace(
            generation_config=GenerationConfig(
                do_sample=True, temperature=1.0, top_k=50, max_new_tokens=16
            ),
            processing_class=SimpleNamespace(pad_token_id=151643, eos_token_id=151645),
        )
        args = SimpleNamespace(max_completion_length=384)
        with patch.dict(
            os.environ,
            {
                "PATHVLM_GENERATION_TEMPERATURE": "0.9",
                "PATHVLM_GENERATION_TOP_P": "1.0",
                "PATHVLM_GENERATION_TOP_K": "0",
                "PATHVLM_GENERATION_TYPICAL_P": "1.0",
                "PATHVLM_GENERATION_REPETITION_PENALTY": "1.0",
            },
            clear=False,
        ):
            contract = freeze_generation_contract(trainer, args)
        self.assertEqual(contract["temperature"], 0.9)
        self.assertTrue(contract["top_k_disabled"])
        self.assertEqual(contract["max_new_tokens"], 384)
        self.assertEqual(contract["stop_contract"], "first_eos_or_max_new_tokens")

    def test_nonuniform_save_steps(self):
        callback = ExplicitSaveStepsCallback()
        control = SimpleNamespace(should_save=False)
        with patch.dict(os.environ, {"PATHVLM_EXPLICIT_SAVE_STEPS": "10,25,50"}):
            callback.on_step_end(None, SimpleNamespace(global_step=25), control)
        self.assertTrue(control.should_save)

    def test_epoch_snapshot_can_use_a_cumulative_step_offset(self):
        callback = EpochSnapshotCallback()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(output_dir=str(root / "output"))
            state = SimpleNamespace(global_step=50, epoch=0.1, is_world_process_zero=True)
            control = SimpleNamespace()
            with patch.dict(
                os.environ,
                {
                    "PATHVLM_EPOCH_SNAPSHOT_STEPS": "50,100",
                    "PATHVLM_EPOCH_SNAPSHOT_DIR": str(root / "snapshots"),
                    "PATHVLM_SNAPSHOT_STEP_OFFSET": "500",
                },
                clear=False,
            ), patch("grpo_pathmmu.create_model_only_snapshot") as create:
                callback.on_save(args, state, control)
            create.assert_called_once_with(
                root / "output/checkpoint-50",
                root / "snapshots/checkpoint-550",
                global_step=550,
                epoch=0.1,
            )

    def test_resume_scheduler_restart_preserves_optimizer_object(self):
        loaded = []
        optimizer = SimpleNamespace(param_groups=[{"lr": 0.0, "initial_lr": 1e-6}])
        trainer = SimpleNamespace(
            optimizer=optimizer,
            lr_scheduler="exhausted",
            args=SimpleNamespace(lr_scheduler_type="linear"),
            _load_optimizer_and_scheduler=lambda checkpoint: loaded.append(checkpoint),
        )

        with patch.dict(
            os.environ,
            {
                "PATHVLM_RESET_SCHEDULER_ON_RESUME_STEPS": "500",
                "PATHVLM_RESET_SCHEDULER_PEAK_LR": "5e-7",
            },
            clear=False,
        ):
            report = install_scheduler_restart_on_resume(trainer)
            trainer._load_optimizer_and_scheduler("checkpoint-500")
        self.assertEqual(loaded, ["checkpoint-500"])
        self.assertIs(trainer.optimizer, optimizer)
        self.assertEqual(optimizer.param_groups[0]["lr"], 5e-7)
        self.assertIsInstance(trainer.lr_scheduler, RelativeLinearScheduler)
        self.assertEqual(trainer.lr_scheduler.total_steps, 500)
        self.assertTrue(report["optimizer_state_resumed"])

    def test_relative_scheduler_uses_segment_local_steps(self):
        optimizer = SimpleNamespace(param_groups=[{"lr": 0.0}])
        scheduler = RelativeLinearScheduler(
            optimizer, total_steps=2, peak_lr=5e-7
        )
        self.assertEqual(scheduler.get_last_lr(), [5e-7])
        scheduler.step()
        self.assertEqual(scheduler.get_last_lr(), [2.5e-7])
        scheduler.step()
        self.assertEqual(scheduler.get_last_lr(), [0.0])

    def test_deepspeed_scheduler_is_not_stepped_twice_by_trainer(self):
        optimizer = SimpleNamespace(param_groups=[{"lr": 0.0}])
        engine = SimpleNamespace(lr_scheduler="exhausted")
        trainer = SimpleNamespace(
            optimizer=optimizer,
            lr_scheduler="exhausted",
            deepspeed=engine,
            args=SimpleNamespace(lr_scheduler_type="linear"),
            _load_optimizer_and_scheduler=lambda checkpoint: None,
        )
        with patch.dict(
            os.environ,
            {
                "PATHVLM_RESET_SCHEDULER_ON_RESUME_STEPS": "2",
                "PATHVLM_RESET_SCHEDULER_PEAK_LR": "5e-7",
            },
            clear=False,
        ):
            install_scheduler_restart_on_resume(trainer)
            trainer._load_optimizer_and_scheduler("checkpoint-500")
        self.assertIsInstance(trainer.lr_scheduler, ReadOnlySchedulerProxy)
        engine.lr_scheduler.step()
        trainer.lr_scheduler.step()
        self.assertEqual(trainer.lr_scheduler.get_last_lr(), [2.5e-7])


if __name__ == "__main__":
    unittest.main()
