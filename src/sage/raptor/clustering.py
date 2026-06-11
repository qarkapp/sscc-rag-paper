"""Hierarchical soft clustering for RAPTOR.

This is thin orchestration over mature libraries: dimensionality reduction uses
``umap-learn`` and clustering uses scikit-learn's ``GaussianMixture`` with BIC-based
model selection and soft (multi-membership) assignment. The two-stage scheme first
clusters globally, then refines large clusters locally -- the structure the
reference system uses, expressed through standard components.

Determinism: UMAP is seeded (``random_state``), which makes it single-threaded and
reproducible; GMM is seeded as well.
"""

from __future__ import annotations

import math

import numpy as np

from sage.config.schema import RaptorCfg

__all__ = ["cluster_bic", "hierarchical_cluster", "reduce_dimensions"]


def reduce_dimensions(
    embeddings: np.ndarray, *, n_neighbors: int, target_dim: int, n_epochs: int, seed: int
) -> np.ndarray:
    """Project embeddings to ``target_dim`` with cosine-metric UMAP."""
    import umap

    n = embeddings.shape[0]
    target_dim = max(2, min(target_dim, n - 2))
    n_neighbors = max(2, min(n_neighbors, n - 1))
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=target_dim,
        min_dist=0.1,
        metric="cosine",
        n_epochs=n_epochs,
        random_state=seed,
        verbose=False,
    )
    return np.asarray(reducer.fit_transform(embeddings), dtype=np.float64)


def cluster_bic(
    points: np.ndarray,
    *,
    max_clusters: int,
    covariance: str,
    threshold: float,
    seed: int,
) -> list[list[int]]:
    """Fit GMMs over a range of ``k``, pick the BIC-optimal model, soft-assign.

    Returns a list of clusters, each a list of member indices. Membership is soft:
    an index may belong to multiple clusters whose responsibility exceeds the
    (k-adjusted) threshold.
    """
    from sklearn.mixture import GaussianMixture

    n = points.shape[0]
    upper = max(1, min(max_clusters, n - 1))
    best_gmm: GaussianMixture | None = None
    best_bic = math.inf
    increases = 0
    # Search from k=1 so a coherent group can decline to split (BIC selects it),
    # which prevents the local-refinement stage from shattering homogeneous clusters.
    for k in range(1, upper + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance,
            random_state=seed,
            reg_covar=1e-6,
            max_iter=100,
            tol=1e-6,
        )
        gmm.fit(points)
        bic = float(gmm.bic(points))
        if bic < best_bic:
            best_bic, best_gmm, increases = bic, gmm, 0
        else:
            increases += 1
            if increases >= 3:  # early stop after 3 consecutive increases
                break

    if best_gmm is None:
        return [list(range(n))]

    responsibilities = best_gmm.predict_proba(points)
    k = responsibilities.shape[1]
    effective = max(threshold, 1.0 / k + 1e-6)
    clusters: list[list[int]] = [[] for _ in range(k)]
    for i in range(n):
        assigned = np.where(responsibilities[i] >= effective)[0]
        if assigned.size == 0:  # fall back to the most likely cluster
            assigned = np.array([int(np.argmax(responsibilities[i]))])
        for c in assigned:
            clusters[int(c)].append(i)
    return [c for c in clusters if c]


def hierarchical_cluster(embeddings: np.ndarray, *, cfg: RaptorCfg, seed: int) -> list[list[int]]:
    """Two-stage clustering: global clusters, then local refinement of large ones.

    Returns clusters as lists of member indices into ``embeddings``.
    """
    n = embeddings.shape[0]
    if n < max(cfg.min_nodes_for_level, cfg.umap_target_dim + 2):
        return [list(range(n))]

    global_neighbors = max(2, math.ceil(math.sqrt(max(n - 1, 1))))
    reduced = reduce_dimensions(
        embeddings,
        n_neighbors=global_neighbors,
        target_dim=cfg.umap_target_dim,
        n_epochs=cfg.umap_n_epochs,
        seed=seed,
    )
    global_clusters = cluster_bic(
        reduced,
        max_clusters=cfg.max_clusters,
        covariance=cfg.cluster_covariance,
        threshold=cfg.soft_cluster_threshold,
        seed=seed,
    )

    refined: list[list[int]] = []
    refine_min = max(cfg.umap_target_dim + 2, cfg.local_refine_min_nodes)
    for members in global_clusters:
        if len(members) <= refine_min:
            refined.append(members)
            continue
        sub_embeddings = embeddings[members]
        local_reduced = reduce_dimensions(
            sub_embeddings,
            n_neighbors=min(10, len(members) - 1),
            target_dim=cfg.umap_target_dim,
            n_epochs=max(1, cfg.umap_n_epochs // 2),
            seed=seed,
        )
        sub_clusters = cluster_bic(
            local_reduced,
            max_clusters=cfg.max_clusters,
            covariance=cfg.cluster_covariance,
            threshold=cfg.soft_cluster_threshold,
            seed=seed,
        )
        for sub in sub_clusters:
            refined.append([members[i] for i in sub])
    return refined
