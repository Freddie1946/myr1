from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_pathmmu_qwen_diagnostic.py")


def test_newer_qwen25_nested_text_config_is_normalized_in_memory():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "AutoConfig.from_pretrained" in source
    assert 'isinstance(getattr(compatible_config, "text_config", None), dict)' in source
    assert 'delattr(compatible_config, "text_config")' in source
    assert 'load_kwargs["config"] = compatible_config' in source
