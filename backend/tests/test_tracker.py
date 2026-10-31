from app.perception.interface import Detection
from app.tracking.tracker import Tracker


def det(x, y, size=40, cls="impure", role="reject", conf=0.9):
    return Detection(0, cls, role, conf, (x, y, x + size, y + size))


def test_stable_id_across_jitter():
    tr = Tracker()
    ids = set()
    for f in range(1, 21):
        tracks = tr.update([det(100 + (f % 3), 100 - (f % 2))], f)
        ids.add(tracks[0].track_id)
    assert ids == {"salt-1"}
    assert tr.tracks["salt-1"].consecutive_hits == 20


def test_two_objects_get_distinct_ids():
    tr = Tracker()
    tracks = tr.update([det(100, 100), det(300, 300)], 1)
    assert {t.track_id for t in tracks} == {"salt-1", "salt-2"}
    tracks = tr.update([det(302, 301), det(101, 99)], 2)  # order swapped
    by_pos = {t.track_id: t.detection.bbox_xyxy_px[0] for t in tracks}
    assert by_pos == {"salt-1": 101, "salt-2": 302}


def test_short_gap_keeps_id_and_restarts_stability():
    tr = Tracker(max_missed=5)
    for f in range(1, 5):
        tr.update([det(100, 100)], f)
    tr.update([], 5)
    tracks = tr.update([det(101, 100)], 6)
    assert tracks[0].track_id == "salt-1"
    assert tracks[0].consecutive_hits == 1


def test_long_disappearance_drops_track():
    tr = Tracker(max_missed=3)
    tr.update([det(100, 100)], 1)
    for f in range(2, 7):
        tr.update([], f)
    assert tr.tracks == {}
    assert tr.update([det(100, 100)], 7)[0].track_id == "salt-2"


def test_class_is_majority_vote():
    tr = Tracker()
    for f in range(1, 6):
        tr.update([det(100, 100)], f)
    tracks = tr.update([det(100, 100, cls="pure", role="accept")], 6)
    assert tracks[0].class_name == "impure" and tracks[0].role == "reject"
