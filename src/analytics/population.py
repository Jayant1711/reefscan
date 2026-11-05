"""Species population statistics — Phase 6 innovation #3."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from ..realtime.schema import FrameAnalysis
from .biodiversity import rarity_report


def species_population_stats(analyses: Sequence[FrameAnalysis], rare_pct: float = 1.0) -> dict:
    """Aggregate per-frame detections into a population table.

    Deduplicates individuals within a frame via track_id when present
    (fallback: one vote per detection), then reports:

    - counts & relative abundance per species
    - dominant / rare species
    - detections per frame (activity profile)
    """
    counter: dict[str, int] = {}
    per_frame_counts: list[int] = []
    for analysis in analyses:
        seen_ids: set[int] = set()
        frame_species: list[str] = []
        for det in analysis.detections:
            if det.track_id is not None:
                if det.track_id in seen_ids:
                    continue
                seen_ids.add(det.track_id)
            frame_species.append(det.species)
        for species in frame_species:
            counter[species] = counter.get(species, 0) + 1
        per_frame_counts.append(len(frame_species))

    total = sum(counter.values())
    rows = [
        {
            "species": species,
            "count": count,
            "relative_abundance_pct": round(100.0 * count / total, 2) if total else 0.0,
        }
        for species, count in sorted(counter.items(), key=lambda kv: -kv[1])
    ]
    rarity = rarity_report(counter, rare_pct)
    return {
        "table": pd.DataFrame(rows),
        "total_detections": total,
        "n_species": len(counter),
        "dominant_species": rarity["dominant"],
        "rare_species": rarity["rare"],
        "mean_detections_per_frame": sum(per_frame_counts) / max(1, len(per_frame_counts)),
        "frames": len(per_frame_counts),
    }
