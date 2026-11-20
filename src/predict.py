import json
import joblib
import numpy as np
import pandas as pd
from src import config, evaluate, features


def main() -> None:
    model = joblib.load(config.MODEL_PATH)
    metadata = json.loads(config.METADATA_PATH.read_text())

    df = pd.read_pickle(config.LABELLED_DATA_PATH)
    open_df = features.add_features(df[df["target"].isna()])

    open_df["enrolment_probability"] = model.predict_proba(features.get_feature_matrix(open_df))[:, 1]
    open_df["priority"] = evaluate.assign_band(open_df["enrolment_probability"], metadata["high_cutoff"], metadata["medium_cutoff"])
    open_df["stage_group"] = np.where(open_df["Most Advanced Stage"].isin(config.LATE_OPEN_STAGES), "Late stage (offer or beyond)", "Early stage")

    id_columns = ["Opportunity ID", "Most Advanced Stage", "stage_group", "enrolment_probability", "priority", "intake_date"]
    scored = open_df[id_columns + config.FEATURES].sort_values("enrolment_probability", ascending=False)

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_csv(config.SCORED_OPEN_PATH, index=False)
    print(f"Scored {len(scored):,} open applications -> {config.SCORED_OPEN_PATH}")
    print(scored["priority"].value_counts().to_dict())


if __name__ == "__main__":
    main()