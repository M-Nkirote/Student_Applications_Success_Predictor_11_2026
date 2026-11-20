"""API and web pages. Prediction tests are skipped until the model has been trained."""
import pytest
from fastapi.testclient import TestClient

from app.auth import DEMO_EMAIL, DEMO_PASSWORD
from app.main import app, service

client = TestClient(app)
needs_model = pytest.mark.skipif(not service.ready, reason="Model not trained yet (run python -m src.pipeline)")

EXAMPLE = {"lead_source": "Referral", "deal_owner": "NEW_AGENT", "student_country": "Kenya",
           "level_of_study": "Postgraduate", "destination": "United Kingdom", "study_type": "On-Campus",
           "intake_month": "Sep", "intake_year": 2026, "created_date": "2026-03-01"}


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_pages_require_login():
    response = TestClient(app).get("/score", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/login"


def test_wrong_password_is_rejected():
    assert client.post("/login", data={"email": DEMO_EMAIL, "password": "wrong"}).status_code == 401


def test_intake_month_needs_year():
    response = client.post("/api/predict", json={**EXAMPLE, "intake_year": None})
    assert response.status_code == 422


@needs_model
def test_api_predict_returns_probability_and_explanation():
    body = client.post("/api/predict", json=EXAMPLE).json()
    assert 0 <= body["probability"] <= 1
    assert body["priority"] in {"High", "Medium", "Low"}
    # SHAP additivity: average + contributions = score
    total = body["average_score"] + sum(item["contribution"] for item in body["contributions"])
    assert abs(total - body["probability"]) < 1e-6


@needs_model
def test_web_scoring_flow():
    web = TestClient(app)
    web.post("/login", data={"email": DEMO_EMAIL, "password": DEMO_PASSWORD})
    response = web.post("/score", data={**EXAMPLE, "intake_year": "2026"})
    assert response.status_code == 200 and "Why this score" in response.text


def first_open_id():
    import pandas as pd
    from src import config
    if not config.SCORED_OPEN_PATH.exists():
        return None
    return pd.read_csv(config.SCORED_OPEN_PATH, dtype={"Opportunity ID": str})["Opportunity ID"].iloc[0]


def test_reference_id_is_echoed_but_not_required():
    if not service.ready:
        pytest.skip("Model not trained yet")
    with_id = client.post("/api/predict", json={**EXAMPLE, "opportunity_id": "REF-1"}).json()
    without_id = client.post("/api/predict", json=EXAMPLE, params={"explain": False}).json()
    assert with_id["opportunity_id"] == "REF-1" and without_id["opportunity_id"] is None
    assert abs(with_id["probability"] - without_id["probability"]) < 1e-12   # the ID never changes the score


@needs_model
def test_open_application_api_and_page():
    opportunity_id = first_open_id()
    if opportunity_id is None:
        pytest.skip("No scored open applications (run python -m src.pipeline)")
    body = client.get(f"/api/applications/{opportunity_id}").json()
    assert body["opportunity_id"] == opportunity_id and 0 <= body["probability"] <= 1
    assert client.get("/api/applications/does-not-exist").status_code == 404

    web = TestClient(app)
    web.post("/login", data={"email": DEMO_EMAIL, "password": DEMO_PASSWORD})
    page = web.get(f"/applications/{opportunity_id}")
    assert page.status_code == 200 and opportunity_id in page.text and "Why this score" in page.text
    search = web.get(f"/pipeline?q={opportunity_id}", follow_redirects=False)
    assert search.status_code == 303 and search.headers["location"] == f"/applications/{opportunity_id}"
