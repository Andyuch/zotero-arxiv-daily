from datetime import datetime

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from paper import ArxivPaper


DEFAULT_MODEL = "avsolatorio/GIST-small-Embedding-v0"
TOP_K = 8
RECENT_CORPUS_SIZE = 75
RECENT_TOP_K = 5
MAX_CLUSTERS = 8
MMR_LAMBDA = 0.85
MMR_POOL_SIZE = 200

# Zero-token local ranking weights. These sum to 1.0.
TOP_K_WEIGHT = 0.50
CLUSTER_WEIGHT = 0.30
RECENT_WEIGHT = 0.20


def _clean_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split())


def _build_text(title: str | None, abstract: str | None) -> str:
    """Build an embedding input that keeps the information-dense title."""
    title = _clean_text(title)
    abstract = _clean_text(abstract)
    if title and abstract:
        return f"Title: {title}\nAbstract: {abstract}"
    return title or abstract


def _date_added(paper: dict) -> datetime:
    value = paper.get("data", {}).get("dateAdded", "")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        return datetime.min


def _top_k_mean(similarity: np.ndarray, k: int) -> np.ndarray:
    """Mean of each candidate's strongest k similarities."""
    if similarity.shape[1] == 0:
        return np.zeros(similarity.shape[0], dtype=np.float32)
    k = min(k, similarity.shape[1])
    top = np.partition(similarity, similarity.shape[1] - k, axis=1)[:, -k:]
    return top.mean(axis=1)


def _cluster_count(corpus_size: int) -> int:
    if corpus_size < 4:
        return 1
    # Roughly one interest cluster per ~20 papers, with conservative bounds.
    return min(MAX_CLUSTERS, max(2, round(np.sqrt(corpus_size / 20))))


def _interest_cluster_score(
    corpus_feature: np.ndarray,
    candidate_feature: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Score candidates against automatically discovered Zotero interests."""
    n_clusters = _cluster_count(len(corpus_feature))
    if n_clusters == 1:
        centroids = corpus_feature.mean(axis=0, keepdims=True)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(corpus_feature)
        centroids = kmeans.cluster_centers_

    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.clip(norms, 1e-12, None)
    score = (candidate_feature @ centroids.T).max(axis=1)
    return score, n_clusters


def _mmr_order(
    base_scores: np.ndarray,
    candidate_feature: np.ndarray,
    lam: float = MMR_LAMBDA,
    pool_size: int = MMR_POOL_SIZE,
) -> list[int]:
    """Diversify only the high-relevance pool using Maximal Marginal Relevance."""
    n = len(base_scores)
    if n <= 1:
        return list(range(n))

    base_order = np.argsort(-base_scores)
    pool = base_order[: min(pool_size, n)]
    tail = base_order[len(pool) :]

    pool_scores = base_scores[pool]
    score_min = float(pool_scores.min())
    score_max = float(pool_scores.max())
    if score_max > score_min:
        relevance = (pool_scores - score_min) / (score_max - score_min)
    else:
        relevance = np.ones_like(pool_scores)

    selected_local: list[int] = [0]
    selected_mask = np.zeros(len(pool), dtype=bool)
    selected_mask[0] = True

    pool_feature = candidate_feature[pool]
    max_similarity = pool_feature @ pool_feature[0]

    while len(selected_local) < len(pool):
        mmr_score = lam * relevance - (1.0 - lam) * max_similarity
        mmr_score[selected_mask] = -np.inf
        next_local = int(np.argmax(mmr_score))
        selected_local.append(next_local)
        selected_mask[next_local] = True
        max_similarity = np.maximum(
            max_similarity,
            pool_feature @ pool_feature[next_local],
        )

    diversified = [int(pool[i]) for i in selected_local]
    diversified.extend(int(i) for i in tail)
    return diversified


def rerank_paper(
    candidate: list[ArxivPaper],
    corpus: list[dict],
    model: str = DEFAULT_MODEL,
) -> list[ArxivPaper]:
    """Rank arXiv candidates using only local embeddings (zero API tokens).

    The score combines:
      1. similarity to the most relevant Zotero papers,
      2. similarity to automatically discovered interest clusters,
      3. similarity to recently added Zotero papers.

    A final MMR pass diversifies only the high-relevance pool so one narrow
    topic cannot dominate the daily digest. The function signature is kept
    backward-compatible with the original recommender.
    """
    if not candidate:
        return candidate
    if not corpus:
        for paper in candidate:
            paper.score = 0.0
        return candidate

    corpus = sorted(corpus, key=_date_added, reverse=True)

    corpus_text = [
        _build_text(
            paper.get("data", {}).get("title", ""),
            paper.get("data", {}).get("abstractNote", ""),
        )
        for paper in corpus
    ]
    candidate_text = [_build_text(paper.title, paper.summary) for paper in candidate]

    encoder = SentenceTransformer(model)
    corpus_feature = encoder.encode(
        corpus_text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    candidate_feature = encoder.encode(
        candidate_text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    similarity = candidate_feature @ corpus_feature.T

    top_k_score = _top_k_mean(similarity, TOP_K)

    cluster_score, n_clusters = _interest_cluster_score(
        corpus_feature,
        candidate_feature,
    )

    recent_size = min(RECENT_CORPUS_SIZE, len(corpus))
    recent_similarity = candidate_feature @ corpus_feature[:recent_size].T
    recent_score = _top_k_mean(recent_similarity, RECENT_TOP_K)

    base_score = (
        TOP_K_WEIGHT * top_k_score
        + CLUSTER_WEIGHT * cluster_score
        + RECENT_WEIGHT * recent_score
    )

    # Keep the familiar ~0-10 score scale used by the original recommender.
    scaled_score = base_score * 10.0
    for i, paper in enumerate(candidate):
        paper.score = float(scaled_score[i])
        # Useful for future debugging/explainability; no LLM or API is involved.
        paper.score_components = {
            "top_k": float(top_k_score[i]),
            "cluster": float(cluster_score[i]),
            "recent": float(recent_score[i]),
        }

    order = _mmr_order(base_score, candidate_feature)

    logger.info(
        "Recommender V2: zero-token local ranking; {} candidates, {} Zotero papers, "
        "{} interest clusters, top-k={}, recent-corpus={}, MMR pool={}",
        len(candidate),
        len(corpus),
        n_clusters,
        min(TOP_K, len(corpus)),
        recent_size,
        min(MMR_POOL_SIZE, len(candidate)),
    )

    return [candidate[i] for i in order]
