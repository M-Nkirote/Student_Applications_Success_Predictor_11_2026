"""SHAP explanations: why did the model give this application its score?

We explain the WHOLE pipeline (preprocessing + model) on the original features, so each contribution belongs to something an agent recognises (lead source, destination, ...), not to a one-hot column.
Contributions are in probability points: average score + sum of contributions = this application's score.

SHAP's masker needs numbers, so categories are temporarily turned into integer codes and turned back into text before the model is called.
"""
import numpy as np
import pandas as pd
import shap

from src import config


class ScoreExplainer:
    def __init__(self, model, background: pd.DataFrame):
        self.model = model
        self.background = background[config.FEATURES].reset_index(drop=True)

    def _categories(self, row: pd.DataFrame) -> dict:
        """All category values seen in the background data or in the row being explained."""
        return {column: sorted(set(self.background[column].dropna().astype(str)) | set(row[column].dropna().astype(str)))
                for column in config.CATEGORICAL_FEATURES}

    @staticmethod
    def _encode(df: pd.DataFrame, categories: dict) -> np.ndarray:
        coded = df[config.FEATURES].copy()
        for column in config.CATEGORICAL_FEATURES:
            coded[column] = pd.Categorical(coded[column].astype("object"), categories=categories[column]).codes
        return coded.astype(float).to_numpy()

    @staticmethod
    def _decode(array: np.ndarray, categories: dict) -> pd.DataFrame:
        df = pd.DataFrame(array, columns=config.FEATURES)
        for column in config.CATEGORICAL_FEATURES:
            df[column] = [categories[column][int(code)] if code >= 0 else np.nan for code in df[column]]
        return df

    def explain(self, row: pd.DataFrame) -> dict:
        """Explain a single application (a one-row DataFrame with the model features)."""
        categories = self._categories(row)

        def predict(array):
            return self.model.predict_proba(self._decode(array, categories))[:, 1]

        background = self._encode(self.background, categories)
        explainer = shap.PermutationExplainer(predict, shap.maskers.Independent(background, max_samples=len(background)),
                                              feature_names=config.FEATURES, seed=config.RANDOM_STATE)
        result = explainer(self._encode(row, categories))

        contributions = []
        for feature, contribution in zip(config.FEATURES, result.values[0]):
            value = row[feature].iloc[0]
            if feature == "intake_lead_months":
                value = "Not known" if pd.isna(value) else f"{value:.1f}"
            contributions.append({
                "feature": feature,
                "label": config.FEATURE_LABELS[feature],
                "value": "Not known" if pd.isna(value) else str(value),
                "contribution": float(contribution),
            })
        contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)

        base_value = float(result.base_values[0])
        return {"average_score": base_value,
                "score": base_value + float(result.values[0].sum()),
                "contributions": contributions}
