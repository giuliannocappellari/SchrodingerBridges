from scripts.nds_common import TRACKS


def test_formal_stop_supports_every_campaign_track():
    assert set(TRACKS) == {"N1", "N2", "N3", "N4", "N5", "N6"}
