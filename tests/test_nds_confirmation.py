from scripts.report_nds_confirmation import relative_reduction


def test_confirmation_relative_reduction():
    assert relative_reduction(0.05, 0.10) == 0.5
    assert relative_reduction(0.0, 0.0) == 1.0
