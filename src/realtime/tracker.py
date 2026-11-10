"""Lightweight centroid tracker for unique fish counting (no heavy MOT deps)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..segmentation.postprocess import BBox


@dataclass
class Track:
    track_id: int
    centroid: tuple[float, float]
    disappeared: int = 0
    species: str = "unknown"
    confidence: float = 0.0
    first_frame: int = 0
    last_frame: int = 0


class CentroidTracker:
    """Nearest-centroid tracker with a disappearance buffer.

    Good enough for unique-species counting in survey video; deliberately
    simple so it runs inside the real-time loop at negligible cost.
    """

    def __init__(self, max_disappeared: int = 15, min_distance: float = 40.0) -> None:
        self.max_disappeared = int(max_disappeared)
        self.min_distance = float(min_distance)
        self.tracks: dict[int, Track] = {}
        # Species of tracks that left the frame — kept so end-of-video
        # statistics count every individual ever seen, not just live ones.
        self.retired_counts: dict[str, int] = {}
        self._next_id = 1

    def update(self, detections: list[BBox], frame_idx: int = 0,
               species_by_box: list[tuple[str, float]] | None = None) -> list[Track]:
        """Update tracks from current detections; returns live tracks."""
        if not detections:
            # Age + retire stale tracks even on empty frames, then return.
            self._drop_stale()
            return list(self.tracks.values())

        centroids = [(d.cx, d.cy) for d in detections]

        # Age existing tracks.
        for track in self.tracks.values():
            track.disappeared += 1

        unmatched_dets = set(range(len(centroids)))
        if self.tracks:
            track_ids = list(self.tracks.keys())
            dist_matrix = np.array([
                [np.hypot(cx - self.tracks[tid].centroid[0], cy - self.tracks[tid].centroid[1])
                 for cx, cy in centroids] for tid in track_ids
            ])
            # Greedy nearest assignment (few simultaneous fish in practice).
            for _ in range(min(len(track_ids), len(centroids))):
                if dist_matrix.size == 0:
                    break
                masked = np.where(dist_matrix <= self.min_distance, dist_matrix, np.nan)
                if not np.isfinite(masked).any():
                    break  # every remaining pair is beyond min_distance
                i, j = np.unravel_index(np.nanargmin(masked), masked.shape)
                tid = track_ids[i]
                track = self.tracks[tid]
                track.centroid = centroids[j]
                track.disappeared = 0
                track.last_frame = frame_idx
                if species_by_box is not None and j < len(species_by_box):
                    track.species, track.confidence = species_by_box[j]
                unmatched_dets.discard(j)
                dist_matrix[i, :] = np.inf
                dist_matrix[:, j] = np.inf

        # Register new tracks for unmatched detections.
        for j in unmatched_dets:
            species, conf = ("unknown", 0.0)
            if species_by_box is not None and j < len(species_by_box):
                species, conf = species_by_box[j]
            self.tracks[self._next_id] = Track(
                track_id=self._next_id, centroid=centroids[j], disappeared=0,
                species=species, confidence=conf, first_frame=frame_idx, last_frame=frame_idx,
            )
            self._next_id += 1

        self._drop_stale()
        return list(self.tracks.values())

    def _drop_stale(self) -> None:
        """Remove stale tracks, remembering their species for final counts."""
        dead = [tid for tid, t in self.tracks.items() if t.disappeared > self.max_disappeared]
        for tid in dead:
            track = self.tracks.pop(tid)
            self.retired_counts[track.species] = self.retired_counts.get(track.species, 0) + 1

    @property
    def unique_count(self) -> int:
        """Total distinct individuals ever registered."""
        return self._next_id - 1

    def species_counts(self) -> dict[str, int]:
        """Unique tracks per species (population statistics input).

        Includes retired (left-the-frame) tracks so end-of-video statistics
        count every individual, not just the ones still on screen.
        """
        counts = dict(self.retired_counts)
        for track in self.tracks.values():
            counts[track.species] = counts.get(track.species, 0) + 1
        return counts
