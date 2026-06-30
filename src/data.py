"""data.py — Synthetic resume/job-description text corpus for ResumeMatch.

Each job description has a pool of candidate resumes with graded
relevance. Resumes are TEXT strings (not tabular features), which
is fundamentally different from JobMatch's feature-based approach.
"""
from __future__ import annotations
import numpy as np
from typing import Any

_SKILL_WORDS = ["python", "sql", "machine", "learning", "deep", "neural", "statistics",
                "data", "analysis", "visualization", "pandas", "numpy", "scikit",
                "tensorflow", "pytorch", "nlp", "computer", "vision", "spark", "aws"]
_ROLE_WORDS = ["engineer", "scientist", "analyst", "developer", "architect",
               "manager", "lead", "senior", "junior", "intern"]
_DOMAIN_WORDS = ["finance", "healthcare", "retail", "marketing", "operations",
                 "research", "product", "platform"]


def _gen_resume(rng, skills, role, domain, years):
    """Generate a realistic resume text from skill/role/domain components."""
    n_skills = rng.integers(3, 8)
    selected = list(rng.choice(skills, min(n_skills, len(skills)), replace=False))
    text = f"Experienced {role} with {years} years in {domain}. "
    text += f"Skilled in {' '.join(selected[:4])}. "
    if years > 5:
        text += f"Led teams and delivered {' '.join(rng.choice(_DOMAIN_WORDS, 2))} projects. "
    text += f"Strong background in {' '.join(selected[4:] or selected[:2])} and statistics."
    return text


def _gen_job(rng, skills, role, domain, req_years):
    """Generate a job description text."""
    n_req = rng.integers(4, 7)
    req = list(rng.choice(skills, min(n_req, len(skills)), replace=False))
    text = f"Looking for a {role} in our {domain} team. "
    text += f"Requires {req_years}+ years experience with {' '.join(req[:4])}. "
    text += f"Knowledge of {' '.join(req[4:] or req[:2])} is a plus. "
    text += f"Must have strong {' '.join(rng.choice(_SKILL_WORDS, 2))} skills."
    return text


def make_synthetic(n_jobs: int = 20, resumes_per_job: int = 15, seed: int = 42) -> dict[str, Any]:
    """Generate resume-job text pairs with graded relevance."""
    rng = np.random.default_rng(seed)
    jobs = []
    resumes = []
    pairs = []

    for job_id in range(n_jobs):
        job_skills = list(rng.choice(_SKILL_WORDS, 6, replace=False))
        job_role = rng.choice(_ROLE_WORDS)
        job_domain = rng.choice(_DOMAIN_WORDS)
        job_years = int(rng.integers(2, 8))
        job_text = _gen_job(rng, job_skills, job_role, job_domain, job_years)
        jobs.append({"id": f"J{job_id:03d}", "text": job_text,
                      "skills": job_skills, "role": job_role,
                      "domain": job_domain, "req_years": job_years})

        for res_id in range(resumes_per_job):
            # Resume may or may not match the job
            overlap = rng.integers(0, 6)
            res_skills = list(set(list(job_skills[:overlap]) +
                                  list(rng.choice(_SKILL_WORDS, 6 - overlap, replace=False))))
            res_role = job_role if rng.random() > 0.3 else rng.choice(_ROLE_WORDS)
            res_domain = job_domain if rng.random() > 0.4 else rng.choice(_DOMAIN_WORDS)
            res_years = int(rng.integers(0, 12))
            res_text = _gen_resume(rng, res_skills, res_role, res_domain, res_years)

            # Grade relevance 0-4 based on match
            skill_match = len(set(job_skills) & set(res_skills)) / len(job_skills)
            role_match = 1.0 if res_role == job_role else 0.0
            domain_match = 1.0 if res_domain == job_domain else 0.0
            years_ok = 1.0 if res_years >= job_years else res_years / job_years
            score = 0.4 * skill_match + 0.2 * role_match + 0.2 * domain_match + 0.2 * years_ok
            relevance = int(min(4, score * 5))

            resume_id = f"R{job_id:03d}_{res_id:02d}"
            resumes.append({"id": resume_id, "text": res_text, "job_id": f"J{job_id:03d}"})
            pairs.append({"job_id": f"J{job_id:03d}", "resume_id": resume_id,
                          "relevance": relevance})

    return {
        "jobs": jobs, "resumes": resumes, "pairs": pairs,
        "n_jobs": n_jobs, "n_resumes": len(resumes), "n_pairs": len(pairs),
    }