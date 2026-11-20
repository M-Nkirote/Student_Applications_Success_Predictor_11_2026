import json
import joblib
import pandas as pd
from src import cleaning, config, evaluate, features
from src.explain import ScoreExplainer


class ModelService:
    def __init__(self):
        self.ready = False
        self.error = None
        try:
            self.model = joblib.load(config.MODEL_PATH)
            self.metadata = json.loads(config.METADATA_PATH.read_text())
            self.explainer = ScoreExplainer(self.model, pd.read_pickle(config.SHAP_BACKGROUND_PATH))
            self.ready = True
        except FileNotFoundError as missing:
            self.error = f"Model files not found ({missing.filename}). Run the training pipeline first: python -m src.pipeline"

    @staticmethod
    def build_features(application: dict) -> pd.DataFrame:
        """application: lead_source, deal_owner, student_country, level_of_study, destination,
        intake_month ('Sep' or 'Undecided'), intake_year, created_date."""
        if application["intake_month"] != "Undecided" and application.get("intake_year"):
            intake_text = f"{application['intake_month']}-{str(application['intake_year'])[-2:]}"
        else:
            intake_text = "Undecided"

        row = pd.DataFrame({
            "Created At": [pd.Timestamp(application["created_date"])],
            "intake_date": cleaning.parse_intake(pd.Series([intake_text])),
            "Lead Source Main Category": [application["lead_source"]],
            "Deal Owner": [application["deal_owner"]],
            "Student Country Location": cleaning.clean_country(pd.Series([application["student_country"]])),
            "level_of_study": cleaning.clean_level_of_study(pd.Series([application["level_of_study"]])),
            "destination": cleaning.clean_destination(pd.Series([application["destination"] or None])),
            "Study Type": [application.get("study_type") or "Undecided"],
        })
        return features.get_feature_matrix(features.add_features(row))

    def score(self, application: dict, explain: bool = True) -> dict:
        """Score an application described by the form or the JSON API."""
        return self.score_row(self.build_features(application), explain)

    def find_open_application(self, opportunity_id: str):
        """Look up an open application in the last pipeline run. Returns (feature row, details) or None."""
        if not config.SCORED_OPEN_PATH.exists():
            return None
        scored = pd.read_csv(config.SCORED_OPEN_PATH, dtype={"Opportunity ID": str})
        if not set(config.FEATURES).issubset(scored.columns):
            return None  
        match = scored[scored["Opportunity ID"] == opportunity_id.strip()]
        if match.empty:
            return None
        record = match.iloc[0]
        details = {
            "opportunity_id": record["Opportunity ID"],
            "stage": record["Most Advanced Stage"],
            "late_stage": record["stage_group"].startswith("Late"),
            "lead_source": record["Lead Source Main Category"],
            "deal_owner": record["Deal Owner"],
            "student_country": record["Student Country Location"] if pd.notna(record["Student Country Location"]) else "Not known",
            "level_of_study": record["level_of_study"] if pd.notna(record["level_of_study"]) else "Not known",
            "destination": record["destination"],
            "study_type": record["Study Type"] if pd.notna(record["Study Type"]) else "Not known",
            "intake_month": pd.Timestamp(record["intake_date"]).strftime("%b") if pd.notna(record["intake_date"]) else "Undecided",
            "intake_year": pd.Timestamp(record["intake_date"]).year if pd.notna(record["intake_date"]) else None,
        }
        return match[config.FEATURES].reset_index(drop=True), details

    def score_row(self, row: pd.DataFrame, explain: bool = True) -> dict:
        """Score (and explain) one row of model features."""
        probability = float(self.model.predict_proba(row)[:, 1][0])
        result = {
            "probability": probability,
            "priority": str(evaluate.assign_band([probability], self.metadata["high_cutoff"], self.metadata["medium_cutoff"])[0]),
            "likely_to_enrol": probability >= self.metadata["threshold"],
        }
        if explain:
            explanation = self.explainer.explain(row)
            result["average_score"] = explanation["average_score"]
            result["contributions"] = explanation["contributions"]
        return result
