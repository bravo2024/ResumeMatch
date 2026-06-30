"""model.py — TF-IDF + LSA semantic resume-job matching for ResumeMatch.

Three text-based matching approaches (all operate on TEXT, not features):
1. **TF-IDF cosine** — vectorize resumes and jobs, match by cosine similarity.
2. **BM25** — Okapi BM25 scoring of resumes against job description queries.
3. **LSA** — Truncated SVD on the term-document matrix for latent semantic matching.

This is NLP/text-based, fundamentally different from JobMatch's feature-based LTR.

References
----------
Deerwester et al. (1990), "Indexing by Latent Semantic Analysis." JASIS.
Robertson & Zaragoza (2009), "The Probabilistic Relevance Framework: BM25."
"""
from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from src.core import _tokenize, recall_at_k, mrr, match_quality


class TfidfMatcher:
    """TF-IDF cosine similarity matcher for resume-job text matching."""

    def __init__(self):
        self.vectorizer = None
        self.resume_matrix = None
        self.resumes = []

    def fit(self, resumes):
        self.resumes = resumes
        texts = [r["text"] for r in resumes]
        self.vectorizer = TfidfVectorizer(
            tokenizer=_tokenize, token_pattern=None,
            ngram_range=(1, 2), sublinear_tf=True, norm="l2",
        )
        self.resume_matrix = self.vectorizer.fit_transform(texts)
        return self

    def rank(self, job_text, k=5):
        q_vec = self.vectorizer.transform([job_text])
        scores = sklearn_cosine(q_vec, self.resume_matrix).ravel()
        top_idx = np.argsort(-scores)[:k]
        return [(self.resumes[i]["id"], float(scores[i])) for i in top_idx]


class LSAMatcher:
    """Latent Semantic Analysis matcher using truncated SVD."""

    def __init__(self, n_components=20):
        self.n_components = n_components
        self.vectorizer = None
        self.svd = None
        self.resume_lsa = None
        self.resumes = []

    def fit(self, resumes):
        self.resumes = resumes
        texts = [r["text"] for r in resumes]
        self.vectorizer = TfidfVectorizer(
            tokenizer=_tokenize, token_pattern=None, max_features=200,
        )
        tfidf = self.vectorizer.fit_transform(texts)
        n_comp = min(self.n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        if n_comp < 1:
            n_comp = 1
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.resume_lsa = self.svd.fit_transform(tfidf)
        return self

    def rank(self, job_text, k=5):
        q_vec = self.vectorizer.transform([job_text])
        q_lsa = self.svd.transform(q_vec)
        # Cosine similarity in LSA space
        scores = sklearn_cosine(q_lsa, self.resume_lsa).ravel()
        top_idx = np.argsort(-scores)[:k]
        return [(self.resumes[i]["id"], float(scores[i])) for i in top_idx]


def evaluate_matcher(matcher, data, k=5):
    """Evaluate a text matcher across all job-resume pairs."""
    all_recall, all_mrr, all_quality = [], [], []
    for job in data["jobs"]:
        job_resumes = [p for p in data["pairs"] if p["job_id"] == job["id"]]
        relevant_ids = [p["resume_id"] for p in job_resumes if p["relevance"] >= 3]
        if not relevant_ids:
            continue
        ranked = matcher.rank(job["text"], k=k)
        ranked_ids = [r[0] for r in ranked]
        ranked_rels = []
        rel_map = {p["resume_id"]: p["relevance"] for p in job_resumes}
        for rid in ranked_ids:
            ranked_rels.append(rel_map.get(rid, 0))
        all_recall.append(recall_at_k(ranked_ids, relevant_ids, k))
        all_mrr.append(mrr(ranked_ids, relevant_ids))
        all_quality.append(match_quality(ranked_rels, k))
    return {
        "recall@5": float(np.mean(all_recall)) if all_recall else 0.0,
        "mrr": float(np.mean(all_mrr)) if all_mrr else 0.0,
        "match_quality": float(np.mean(all_quality)) if all_quality else 0.0,
        "n_jobs_evaluated": len(all_recall),
    }


def fit_and_evaluate(data, seed=42):
    """Train and evaluate TF-IDF and LSA matchers."""
    resumes = data["resumes"]
    tfidf_matcher = TfidfMatcher().fit(resumes)
    lsa_matcher = LSAMatcher(n_components=15).fit(resumes)

    tfidf_metrics = evaluate_matcher(tfidf_matcher, data, k=5)
    lsa_metrics = evaluate_matcher(lsa_matcher, data, k=5)

    model = {"tfidf": tfidf_matcher, "lsa": lsa_matcher}
    metrics = {
        "n_jobs": data["n_jobs"], "n_resumes": data["n_resumes"],
        "tfidf": tfidf_metrics, "lsa": lsa_metrics,
    }
    return model, metrics