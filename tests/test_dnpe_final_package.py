from __future__ import annotations

from scripts.finalize_dnpe_campaign import REQUIRED_PACKAGE_FILES


def test_terminal_package_required_files_match_protocol() -> None:
    required = set(REQUIRED_PACKAGE_FILES)
    assert "report_summary.json" in required
    assert "terminal_package_validation.json" in required
    assert "causal_heatmap.png" in required
    assert "reproducibility_manifest.json" in required
    assert len(required) == 19
