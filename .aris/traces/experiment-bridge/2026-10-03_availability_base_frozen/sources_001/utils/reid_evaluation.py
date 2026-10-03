"""Official RGBNT/Market-1501 style person ReID evaluation."""

from __future__ import annotations

import numpy as np


def evaluate_reid(
    distances: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    query_cameras: np.ndarray,
    gallery_cameras: np.ndarray,
    *,
    max_rank: int = 50,
) -> tuple[np.ndarray, float]:
    """Return CMC and mAP after same-identity/same-camera exclusion."""

    distances = np.asarray(distances)
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cameras = np.asarray(query_cameras)
    gallery_cameras = np.asarray(gallery_cameras)

    num_queries, num_gallery = distances.shape
    effective_rank = min(max_rank, num_gallery)
    ranked_gallery = np.argsort(distances, axis=1, kind="stable")
    query_curves: list[np.ndarray] = []
    query_average_precisions: list[float] = []

    for query_index in range(num_queries):
        order = ranked_gallery[query_index]
        junk = (gallery_ids[order] == query_ids[query_index]) & (
            gallery_cameras[order] == query_cameras[query_index]
        )
        matches = gallery_ids[order][~junk] == query_ids[query_index]
        if not np.any(matches):
            continue

        cumulative_matches = np.cumsum(matches)
        cmc = np.minimum(cumulative_matches, 1).astype(np.float64)
        if cmc.size < effective_rank:
            cmc = np.pad(cmc, (0, effective_rank - cmc.size), mode="edge")
        query_curves.append(cmc[:effective_rank])

        precision_at_rank = cumulative_matches / np.arange(1, matches.size + 1)
        average_precision = float(precision_at_rank[matches].mean())
        query_average_precisions.append(average_precision)

    if not query_curves:
        raise RuntimeError("no valid query identity appears in the gallery")

    return (
        np.mean(np.stack(query_curves), axis=0),
        float(np.mean(query_average_precisions)),
    )
