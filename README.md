# ResumeMatch

> AI-powered resume screening and job matching with TF-IDF semantic alignment.

Trains four classifiers on synthetic resume data to predict shortlist eligibility. Dashboard provides data exploration, multi-model comparison, resume quality analysis by keyword match score and format quality distribution, TF-IDF cosine similarity between resumes and job descriptions, Cohen's Kappa inter-rater agreement analysis, and weighted composite candidate scoring.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

Best model (Random Forest) holdout results:

| Metric | Value |
|---|---|
| ROC AUC | 0.778 |
| Gini | 0.556 |
| KS Statistic | 0.463 |
| F1 Score | 0.562 |
| Accuracy | 0.730 |

5-fold CV AUC: 0.766 ± 0.022. Four models compared.

## Features

| Tab | What it does |
|---|---|
| **Explorer** | Resume dataset overview, shortlist rate, outcome distribution |
| **Model Lab** | Multi-model comparison, ROC/calibration curves, CV results, confusion matrix |
| **Resume Quality** | Keyword match score and format quality histograms by shortlist outcome |
| **Screening** | TF-IDF similarity (resume vs job description), composite candidate score, rank shortlist |

## Repo Structure

```
ResumeMatch/
  src/         data, model, evaluate, persist, visualizations modules
  train.py     training pipeline (multi-model + CV)
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Synthetic resume dataset: keyword match score, format quality, experience years, education level, certification count, skills diversity, and shortlist label.

## License

MIT
