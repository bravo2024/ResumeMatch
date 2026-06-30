"""Smoke tests for ResumeMatch — NLP semantic text matching."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import make_synthetic
from src.model import TfidfMatcher, LSAMatcher, fit_and_evaluate, evaluate_matcher
from src.core import cosine_similarity, recall_at_k, mrr, jaccard_similarity


def test_data():
    d = make_synthetic(n_jobs=10, resumes_per_job=8, seed=42)
    assert d["n_jobs"] == 10
    assert d["n_resumes"] == 80
    assert all("text" in j for j in d["jobs"])
    assert all("text" in r for r in d["resumes"])


def test_cosine_similarity():
    v1 = np.array([1, 0, 0])
    v2 = np.array([1, 0, 0])
    assert abs(cosine_similarity(v1, v2) - 1.0) < 0.01


def test_jaccard():
    assert abs(jaccard_similarity(["a", "b", "c"], ["b", "c", "d"]) - 0.5) < 0.01


def test_tfidf_matcher():
    """TF-IDF matcher ranks resumes by cosine similarity to job text."""
    d = make_synthetic(n_jobs=10, resumes_per_job=8, seed=42)
    matcher = TfidfMatcher().fit(d["resumes"])
    ranked = matcher.rank(d["jobs"][0]["text"], k=5)
    assert len(ranked) == 5
    assert all(isinstance(r[0], str) for r in ranked)


def test_lsa_matcher():
    """LSA matcher uses truncated SVD for latent semantic matching."""
    d = make_synthetic(n_jobs=10, resumes_per_job=8, seed=42)
    matcher = LSAMatcher(n_components=5).fit(d["resumes"])
    ranked = matcher.rank(d["jobs"][0]["text"], k=5)
    assert len(ranked) == 5


def test_fit_and_evaluate():
    """Full pipeline evaluates both TF-IDF and LSA matchers."""
    d = make_synthetic(n_jobs=15, resumes_per_job=8, seed=42)
    model, metrics = fit_and_evaluate(d, seed=42)
    assert "tfidf" in metrics
    assert "lsa" in metrics
    assert metrics["tfidf"]["recall@5"] >= 0.0


if __name__ == "__main__":
    test_data()
    test_cosine_similarity()
    test_jaccard()
    test_tfidf_matcher()
    test_lsa_matcher()
    test_fit_and_evaluate()
    print("All ResumeMatch smoke tests passed!")
