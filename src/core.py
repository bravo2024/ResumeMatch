"""core.py — Semantic text matching metrics for ResumeMatch (InfoEdge).

Implements text-matching/ranking metrics, NOT generic classification:
  * **recall@k** — fraction of relevant resumes in top-k.
  * **MRR** — reciprocal rank of first relevant resume.
  * **cosine_similarity** — vector similarity for embeddings.
  * **match_quality** — weighted score combining rank and relevance.

Different from JobMatch (feature-based LTR): ResumeMatch operates on
TEXT (resume and job description strings), not tabular features.

References
----------
Deerwester et al. (1990), "Indexing by Latent Semantic Analysis."
Robertson & Zaragoza (2009), "The Probabilistic Relevance Framework."
"""
from __future__ import annotations
import re
import numpy as np
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset("a an the and or of to in on for is are was were be with as at by it its from this that to you we our".split())


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2 and t not in _STOPWORDS]


def cosine_similarity(v1, v2) -> float:
    """Cosine similarity between two vectors."""
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / norm) if norm > 0 else 0.0


def recall_at_k(ranked_ids, relevant_ids, k):
    relevant = set(relevant_ids)
    if not relevant or k <= 0:
        return 0.0
    return sum(1 for r in list(ranked_ids)[:k] if r in relevant) / len(relevant)


def mrr(ranked_ids, relevant_ids):
    relevant = set(relevant_ids)
    for i, rid in enumerate(ranked_ids, 1):
        if rid in relevant:
            return 1.0 / i
    return 0.0


def match_quality(ranked_relevances, k=5):
    """Weighted match quality: sum(rel_i / log2(i+1)) / max_possible."""
    r = np.asarray(ranked_relevances, dtype=float)[:k]
    if r.size == 0:
        return 0.0
    weights = 1.0 / np.log2(np.arange(2, r.size + 2))
    return float(np.sum(r * weights) / np.sum(weights * 4.0))  # normalize by max grade 4


def jaccard_similarity(set1, set2) -> float:
    """Jaccard similarity between two token sets."""
    s1, s2 = set(set1), set(set2)
    union = s1 | s2
    return len(s1 & s2) / len(union) if union else 0.0