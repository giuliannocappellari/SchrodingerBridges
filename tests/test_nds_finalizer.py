from scripts.finalize_next_direction_selection import CLASS_PRIORITY, REQUIRED_FILES


def test_frozen_selection_class_order_and_package_contract():
    assert CLASS_PRIORITY["A"] > CLASS_PRIORITY["B"] > CLASS_PRIORITY["C"] > CLASS_PRIORITY["D"] > CLASS_PRIORITY["E"]
    assert "next_direction_recommendation.md" in REQUIRED_FILES
    assert "SELECTED_DIRECTION_FULL_CAMPAIGN_DRAFT.md" in REQUIRED_FILES
    assert "terminal_package_validation.json" in REQUIRED_FILES
