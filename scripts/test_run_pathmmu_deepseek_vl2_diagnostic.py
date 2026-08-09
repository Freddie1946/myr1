import ast
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_pathmmu_deepseek_vl2_diagnostic.py")


def constant(name):
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing constant: {name}")


def test_letter_only_contract_is_unambiguous():
    prompt = constant("LETTER_ONLY_TEMPLATE").format(
        question="Question? Options: A) x B) y C) z D) w"
    )
    assert "exactly one uppercase option letter" in prompt
    assert prompt.startswith("<image>\n")


def test_multi_gpu_memory_cap_is_explicit():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert '"device_map": "auto"' in source
    assert '"48GiB"' in source
