from __future__ import annotations

from scripts.run_dnpe_runtime_baseline import prompt_memory_tasks


def test_prompt_memory_uses_only_edit_request_and_prompt() -> None:
    rows = [{"case_id": "a", "rewrite_prompt": "Alice works at", "target_new": "Acme"}]
    tasks = [{"case_id": "a", "prompt": "Alice works at", "bucket": "rewrite"}]
    output = prompt_memory_tasks(tasks, rows)
    assert output[0]["original_prompt"] == "Alice works at"
    assert "Alice works at Acme" in output[0]["prompt"]
    assert "prompt_type" not in output[0]["prompt"]
