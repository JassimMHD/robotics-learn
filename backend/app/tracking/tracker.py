"""Deterministic IoU/centroid tracker giving stable IDs to detections across frames."""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from app.perception.interface import Detection


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: str
    detection: Detection  # latest matched detection
    first_frame: int
    last_frame: int
    hits: int = 1
    consecutive_hits: int = 1
    misses: int = 0
    class_votes: Counter = field(default_factory=Counter)
    class_roles: dict[str, str] = field(default_factory=dict)
    status: str = "observed"  # observed | stable | eligible | ignored | queued | processed_sim | failed_sim
    reason: str = ""

    def observe(self, d: Detection) -> None:
        self.class_votes[d.class_name] += 1
        self.class_roles[d.class_name] = d.role

    @property
    def class_name(self) -> str:
        """Majority class over the track's history (resists single-frame flicker)."""
        return self.class_votes.most_common(1)[0][0]

    @property
    def role(self) -> str:
        return self.class_roles[self.class_name]


class Tracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 15, id_prefix: str = "salt"):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.id_prefix = id_prefix
        self.tracks: dict[str, Track] = {}
        self._next = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next = 1

    def update(self, detections: list[Detection], frame_id: int) -> list[Track]:
        """Associate detections with tracks (greedy by IoU, centroid fallback). Returns tracks seen this frame."""
        tracks = list(self.tracks.values())
        pairs = []
        for ti, t in enumerate(tracks):
            for di, d in enumerate(detections):
                score = iou(t.detection.bbox_xyxy_px, d.bbox_xyxy_px)
                if score < self.iou_threshold:
                    # Centroid fallback: jitter on small boxes can drop IoU below threshold.
                    size = max(t.detection.min_side_px, 1.0)
                    dist = float(np.hypot(*np.subtract(t.detection.center_px, d.center_px)))
                    score = 0.5 * self.iou_threshold if dist < 0.5 * size else 0.0
                if score > 0:
                    pairs.append((score, ti, di))
        pairs.sort(key=lambda p: p[0], reverse=True)
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t, d = tracks[ti], detections[di]
            t.consecutive_hits = t.consecutive_hits + 1 if t.misses == 0 else 1
            t.detection, t.last_frame, t.misses = d, frame_id, 0
            t.hits += 1
            t.observe(d)
        for ti, t in enumerate(tracks):
            if ti not in used_t:
                t.misses += 1
                t.consecutive_hits = 0
        for di, d in enumerate(detections):
            if di not in used_d:
                tid = f"{self.id_prefix}-{self._next}"
                self._next += 1
                track = Track(tid, d, frame_id, frame_id)
                track.observe(d)
                self.tracks[tid] = track
        for tid in [tid for tid, t in self.tracks.items() if t.misses > self.max_missed]:
            del self.tracks[tid]
        return [t for t in self.tracks.values() if t.misses == 0]
