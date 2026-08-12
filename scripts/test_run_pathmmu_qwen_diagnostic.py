from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_pathmmu_qwen_diagnostic.py")


def test_newer_qwen25_nested_text_config_is_normalized_in_memory():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "AutoConfig.from_pretrained" in source
    assert 'isinstance(getattr(compatible_config, "text_config", None), dict)' in source
    assert 'delattr(compatible_config, "text_config")' in source
    assert 'load_kwargs["config"] = compatible_config' in source


def test_optional_peft_adapter_is_loaded_read_only_and_audited():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "from peft import PeftModel" in source
    assert 'parser.add_argument("--adapter", type=Path)' in source
    assert "PeftModel.from_pretrained(" in source
    assert "local_files_only=True, is_trainable=False" in source
    assert '"adapter_config_sha256"' in source
    assert '"adapter_model_sha256"' in source


def test_resume_may_change_only_operational_batch_size_and_records_history():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'ignored_resume_fields = {"resume", "batch_size", "batch_size_history"}' in source
    assert 'existing_config.get("batch_size_history", [existing_config["batch_size"]])' in source
    assert 'run_config["batch_size_history"] = batch_size_history' in source
