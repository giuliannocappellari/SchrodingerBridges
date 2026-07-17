from __future__ import annotations

from scripts.select_dnpe_site_policies import Counter


def test_site_policy_module_imports_without_model_loading() -> None:
    counts = Counter([3, 3, 4])
    assert counts.most_common(1) == [(3, 2)]
