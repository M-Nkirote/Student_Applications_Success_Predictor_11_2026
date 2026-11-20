# Models

Created by `python -m src.train` (or `docker compose run --rm train`):

- `enrolment_model.joblib`: the full scikit-learn pipeline (preprocessing + model)
- `model_metadata.json`: features, thresholds, priority-band cut-offs, test metrics, form options, library versions
- `shap_background.pkl`: 100 training rows used as the SHAP reference

Commit these files: they are the serialized artifacts required by the brief.
