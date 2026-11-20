from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "higher_studies_inc_pseudonymised_data.csv"
INTERIM_DIR = ROOT_DIR / "data" / "interim"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
REPORTS_DIR = ROOT_DIR / "reports"

CLEAN_DATA_PATH = INTERIM_DIR / "clean.pkl"
DATA_INFO_PATH = INTERIM_DIR / "data_info.json"
LABELLED_DATA_PATH = PROCESSED_DIR / "applications.pkl"

MODEL_PATH = MODELS_DIR / "enrolment_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
SHAP_BACKGROUND_PATH = MODELS_DIR / "shap_background.pkl"
SCORED_OPEN_PATH = OUTPUTS_DIR / "open_applications_scored.csv"
SPLIT_PATH = OUTPUTS_DIR / "train_test_split.csv"
METRICS_PATH = REPORTS_DIR / "metrics.json"
LIFT_TABLE_PATH = REPORTS_DIR / "lift_table.csv"

RANDOM_STATE = 42

KEEP_COLUMNS = [
    "Opportunity ID", "Student ID", "Most Advanced Stage", "Created At", "Enrolled Date",
    "Correct Intake", "Lead Source Main Category", "Deal Owner", "Student Country Location",
    "What Level Of Study Are You Applying For", "Study Destination", "Study Type", "Inserted At",
]

# Plausible window for event dates (anything outside is a data-entry error)
DATE_MIN = "2015-01-01"
DATE_MAX_YEARS_AFTER_SNAPSHOT = 1

STAGE_FIXES = {"Enrolled , but no Commission": "Enrolled, but no Commission"}

WON_STAGES = ["Student Enrolled", "Enrolled, but Dropped Out", "Enrolled, but no Commission"]

LOST_STAGES = [
    "Application Withdrawn", "Application Rejected", "Offer Declined",
    "Visa Application Withdrawn", "Visa Application Rejected",
    "Did Not Enroll (after paying deposit)", "Did Not Enroll (after visa issued)",
    "Closed-Lost to Competition", "Not Interested",
]

EARLY_OPEN_STAGES = [
    "Open Opportunity", "Awaiting Documents", "Course Decided",
    "Application Ready for Submission", "Application Done",
]

LATE_OPEN_STAGES = [
    "Conditional Offer Received", "Conditional Offer Accepted", "Offer Received", "Offer Accepted",
    "Visa Application Submitted", "Visa Issued", "Deposit Paid", "Future Intake",
]

STALE_MONTHS = 12 # early-stage open applications older than this are labelled lost
INTAKE_BUFFER_MONTHS = 1 # training window: intake must be this long before the snapshot
MATURE_MONTHS_NO_INTAKE = 12 # ... or, with no intake, the application must be this old

COUNTRY_MAPPING = {
    "Nairobi": "Kenya", "Mombasa": "Kenya", "Eldoret": "Kenya", "Nakuru": "Kenya", "Kitale": "Kenya",
    "Bungoma": "Kenya", "Nanyuki": "Kenya", "Lodwar": "Kenya", "Baringo": "Kenya", "Ziwa": "Kenya",
    "Kenya,Kitale": "Kenya", "Ke": "Kenya", "Ken": "Kenya", "Ke Nya": "Kenya", "Patel": "Kenya",
    "Other": None, "Undecided": None,
    "Kampala": "Uganda", "Jinja": "Uganda", "Harare": "Zimbabwe", "Lagos": "Nigeria",
    "Port Harcourt": "Nigeria", "Dar Es Salaam": "Tanzania", "Dubai": "United Arab Emirates",
    "Dublin Ireland": "Ireland",
    "Uk": "United Kingdom", "Usa": "United States", "Uae": "United Arab Emirates", "Gy": "Guyana",
    "Drc": "DR Congo", "Democratic Republic Of Congo": "DR Congo", "Decocratic Republic Of Congo": "DR Congo",
    "Congo The Democratic Republic Of The": "DR Congo", "Ethopia": "Ethiopia", "Zimbambwe": "Zimbabwe",
    "Botwana": "Botswana", "Somali": "Somalia", "Swaziland": "Eswatini", "Turkey": "Türkiye",
    "Ireland/Poland": "Ireland",
}

LEVEL_MAPPING = {
    "Undergraduate": "Undergraduate", "Bachelor's": "Undergraduate", "Degree": "Undergraduate",
    "Degreee": "Undergraduate", "Undergraduate Diploma/Degree": "Undergraduate",
    "Postgraduate": "Postgraduate", "Masters": "Postgraduate", "Postgraduate Diploma/Degree": "Postgraduate",
    "PHD": "Postgraduate",
    "Diploma": "Diploma/Certificate", "Certificate": "Diploma/Certificate",
    "High School Certificate/Diploma": "Diploma/Certificate", "High School": "Diploma/Certificate",
    "Other": "Other/Undecided", "Undecided": "Other/Undecided",
    "Diploma/Certificate": "Diploma/Certificate", "Other/Undecided": "Other/Undecided",
}

DESTINATION_MAPPING = {
    "England": "United Kingdom", "Dubai": "United Arab Emirates", "Swaziland": "Eswatini",
    "Congo The Democratic Republic Of The": "DR Congo",
}

NUMERIC_FEATURES = ["intake_lead_months"]
CATEGORICAL_FEATURES = [
    "Lead Source Main Category", "Deal Owner", "Student Country Location",
    "level_of_study", "destination", "Study Type",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Plain-language names shown to agents in the web app
FEATURE_LABELS = {
    "intake_lead_months": "Months until intake",
    "Lead Source Main Category": "Lead source",
    "Deal Owner": "Deal owner",
    "Student Country Location": "Student country",
    "level_of_study": "Level of study",
    "destination": "Study destination",
    "Study Type": "Study type",
    "intake_month": "Intake month",
    "created_month": "Month application was created",
}

MIN_FREQUENCY = 20   # rarer categories are grouped into one "infrequent" bucket by the encoder

N_SPLITS = 5
TUNING_ITERATIONS = 40
PR_AUC_TOLERANCE = 0.02          # final model choice (gradient boosting vs logistic regression)
HIGH_BAND_QUANTILE = 0.80        # top 20% of scores = High priority
MEDIUM_BAND_QUANTILE = 0.50      # next 30% = Medium, rest = Low

PARAM_DISTRIBUTIONS = {
    "model__learning_rate": [0.01, 0.02, 0.05, 0.1],
    "model__max_iter": [150, 300, 500, 800, 1000],
    "model__max_leaf_nodes": [4, 8, 15, 31],
    "model__max_depth": [3, 5, None],
    "model__min_samples_leaf": [5, 10, 20, 50, 100],
    "model__l2_regularization": [0.0, 0.1, 1.0],
}

SHAP_BACKGROUND_SIZE = 100
