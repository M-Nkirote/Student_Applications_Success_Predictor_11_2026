from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src import config


def make_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=config.MIN_FREQUENCY,
                                 sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, config.NUMERIC_FEATURES),
        ("cat", categorical_pipe, config.CATEGORICAL_FEATURES),
    ])


def make_gradient_boosting() -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor()),
        ("model", HistGradientBoostingClassifier(random_state=config.RANDOM_STATE)),
    ])


def make_logistic_regression() -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor()),
        ("model", LogisticRegression(max_iter=2000)),
    ])
