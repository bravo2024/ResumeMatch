from __future__ import annotations
import numpy as np; import pandas as pd
FEATURE_NAMES = ["keyword_match_score","format_quality","section_completeness","experience_relevance","education_relevance","certification_count","gap_years","language_proficiency","ats_score","reference_quality","job_title_match_score","salary_fit_score","remote_readiness","industry_experience_diversity"]
CATEGORICAL_FEATURES = ["language_proficiency"]
NUMERICAL_FEATURES = ["keyword_match_score","format_quality","section_completeness","experience_relevance","education_relevance","certification_count","gap_years","ats_score","reference_quality","job_title_match_score","salary_fit_score","remote_readiness","industry_experience_diversity"]
TARGET_NAME = "shortlisted"
def make_synthetic(n=10000,seed=42):
    rng=np.random.default_rng(seed)
    df=pd.DataFrame({
        "keyword_match_score": rng.beta(5,3,size=n).round(3),
        "format_quality": rng.beta(6,2,size=n).round(3),
        "section_completeness": rng.beta(7,2,size=n).round(3),
        "experience_relevance": rng.beta(4,3,size=n).round(3),
        "education_relevance": rng.beta(4,3,size=n).round(3),
        "certification_count": rng.poisson(lam=2,size=n).clip(0,8),
        "gap_years": rng.exponential(scale=1.5,size=n).clip(0,10).round(1),
        "language_proficiency": rng.choice(["basic","intermediate","advanced","native"],size=n,p=[0.10,0.25,0.40,0.25]),
        "ats_score": rng.uniform(0,100,size=n).round(1),
        "reference_quality": rng.uniform(0,100,size=n).round(1),
        "job_title_match_score": rng.beta(4,4,size=n).round(3),
        "salary_fit_score": rng.beta(5,3,size=n).round(3),
        "remote_readiness": rng.beta(6,3,size=n).round(3),
        "industry_experience_diversity": rng.poisson(lam=3,size=n).clip(0,10),
    })
    kw=df["keyword_match_score"]; fmt=df["format_quality"]; sect=df["section_completeness"]
    exp_rel=df["experience_relevance"]; edu_rel=df["education_relevance"]
    cert=np.clip(df["certification_count"]/8,0,1); gap=np.clip(df["gap_years"]/10,0,1)
    lang=df["language_proficiency"].map({"basic":0,"intermediate":0.4,"advanced":0.7,"native":1}).values
    ats=df["ats_score"]/100; ref=df["reference_quality"]/100
    jtm=df["job_title_match_score"]; sfs=df["salary_fit_score"]; remote=df["remote_readiness"]
    ind=np.clip(df["industry_experience_diversity"]/10,0,1)
    log_odds = -1.5 + 0.8*kw + 0.4*fmt + 0.3*sect + 0.6*exp_rel + 0.5*edu_rel + 0.3*cert - 0.4*gap + 0.2*lang + 0.4*ats + 0.3*ref + 0.5*jtm + 0.3*sfs + 0.2*remote + 0.2*ind + rng.normal(0,0.4,size=n)
    prob=1/(1+np.exp(-log_odds)); y=(prob>np.percentile(prob,72)).astype(np.float64)
    return {"X":df,"y":y,"features":FEATURE_NAMES,"df":df.assign(shortlisted=y),"categorical_features":CATEGORICAL_FEATURES,"numerical_features":NUMERICAL_FEATURES,"n_samples":n,"n_features":len(FEATURE_NAMES),"positive_rate":y.mean()}
