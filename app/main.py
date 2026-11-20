import os
from datetime import date
from pathlib import Path
from urllib.parse import quote
from typing import Annotated
import pandas as pd
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from starlette.middleware.sessions import SessionMiddleware
from app.auth import check_credentials, current_agent
from app.model_service import ModelService
from app.schemas import ApplicationInput, ScoreResponse
from src import config

APP_DIR = Path(__file__).resolve().parent
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

app = FastAPI(title="Enrolment outlook API",
              description="Scores study-abroad applications by their probability of ending in an enrolment.")
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "change-this-secret"))
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")

service = ModelService()   
# JSON API

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": service.ready,
            "algorithm": service.metadata["algorithm"] if service.ready else None}


@app.post("/api/predict", response_model=ScoreResponse)
def api_predict(application: ApplicationInput, explain: bool = True):
    if not service.ready:
        raise HTTPException(status_code=503, detail=service.error)
    return {"opportunity_id": application.opportunity_id, **service.score(application.model_dump(), explain=explain)}


@app.get("/api/applications/{opportunity_id}", response_model=ScoreResponse)
def api_open_application(opportunity_id: str, explain: bool = True):
    """Score and explain an open application from the last pipeline run, by its Opportunity ID."""
    if not service.ready:
        raise HTTPException(status_code=503, detail=service.error)
    found = service.find_open_application(opportunity_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"No open application with Opportunity ID {opportunity_id}")
    row, _ = found
    return {"opportunity_id": opportunity_id, **service.score_row(row, explain=explain)}


# Login
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {"error": None})


@app.post("/login", response_class=HTMLResponse)
def login(request: Request, email: Annotated[str, Form()], password: Annotated[str, Form()]):
    if check_credentials(email, password):
        request.session["agent"] = email.strip().lower()
        return RedirectResponse("/score", status_code=303)
    return templates.TemplateResponse(request, "login.html", {"error": "That email and password don't match the demo account."},
                                      status_code=401)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


# Web pages for agents
def form_context(error=None, values=None) -> dict:
    today = date.today()
    return {
        "options": service.metadata["form_options"] if service.ready else {},
        "months": MONTHS,
        "years": list(range(today.year, today.year + 3)),
        "today": today.isoformat(),
        "error": error or service.error,
        "values": values or {},
    }


@app.get("/")
def home(request: Request):
    return RedirectResponse("/score" if current_agent(request) else "/login", status_code=303)


@app.get("/score", response_class=HTMLResponse)
def score_form(request: Request):
    if not current_agent(request):
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(request, "score.html", {"agent": current_agent(request), **form_context()})


@app.post("/score", response_class=HTMLResponse)
def score_application(request: Request,
                      lead_source: Annotated[str, Form()], deal_owner: Annotated[str, Form()],
                      student_country: Annotated[str, Form()], level_of_study: Annotated[str, Form()],
                      destination: Annotated[str, Form()], intake_month: Annotated[str, Form()],
                      created_date: Annotated[str, Form()], intake_year: Annotated[str, Form()] = "",
                      opportunity_id: Annotated[str, Form()] = ""):
    if not current_agent(request):
        return RedirectResponse("/login", status_code=303)

    values = {"lead_source": lead_source, "deal_owner": deal_owner, "student_country": student_country,
              "level_of_study": level_of_study, "destination": destination, "intake_month": intake_month,
              "intake_year": int(intake_year) if intake_year else None, "created_date": created_date,
              "opportunity_id": opportunity_id.strip() or None}
    try:
        application = ApplicationInput(**values)
    except ValidationError:
        message = "Some fields are not valid. Check the dates, and choose a year if you picked an intake month."
        return templates.TemplateResponse(request, "score.html", {"agent": current_agent(request),
                                          **form_context(message, values)},
                                          status_code=422)
    if not service.ready:
        return templates.TemplateResponse(request, "score.html", {"agent": current_agent(request), **form_context()}, status_code=503)

    result = service.score(application.model_dump())
    largest = max(abs(c["contribution"]) for c in result["contributions"]) or 1
    return templates.TemplateResponse(request, "result.html", {
        "agent": current_agent(request), "application": values, "result": result, "largest": largest,
        "metadata": service.metadata,
    })


@app.get("/applications/{opportunity_id}", response_class=HTMLResponse)
def open_application(request: Request, opportunity_id: str):
    """One open application from the last pipeline run: score, priority and SHAP explanation."""
    if not current_agent(request):
        return RedirectResponse("/login", status_code=303)
    found = service.find_open_application(opportunity_id) if service.ready else None
    if found is None:
        return RedirectResponse(f"/pipeline?q={quote(opportunity_id)}", status_code=303)

    row, details = found
    result = service.score_row(row)
    largest = max(abs(c["contribution"]) for c in result["contributions"]) or 1
    return templates.TemplateResponse(request, "result.html", {
        "agent": current_agent(request), "application": details, "result": result, "largest": largest,
        "metadata": service.metadata, "from_pipeline": True,
    })


@app.get("/pipeline", response_class=HTMLResponse)
def open_pipeline(request: Request, owner: str = "", priority: str = "", q: str = ""):
    if not current_agent(request):
        return RedirectResponse("/login", status_code=303)

    rows, owners, can_explain = [], [], False
    q = q.strip()
    if config.SCORED_OPEN_PATH.exists():
        scored = pd.read_csv(config.SCORED_OPEN_PATH, dtype={"Opportunity ID": str})
        owners = sorted(scored["Deal Owner"].dropna().unique())
        # Application pages need the model inputs, which only newer pipeline runs save
        can_explain = service.ready and set(config.FEATURES).issubset(scored.columns)
        if q:
            # An exact ID goes straight to that application's page; part of an ID filters the list
            if can_explain and (scored["Opportunity ID"] == q).any():
                return RedirectResponse(f"/applications/{quote(q)}", status_code=303)
            scored = scored[scored["Opportunity ID"].str.contains(q, regex=False, na=False)]
        if owner:
            scored = scored[scored["Deal Owner"] == owner]
        if priority:
            scored = scored[scored["priority"] == priority]
        rows = scored.head(100).to_dict(orient="records")
    return templates.TemplateResponse(request, "pipeline.html", {
        "agent": current_agent(request), "rows": rows, "owners": owners, "owner": owner, "priority": priority, "q": q,
        "has_file": config.SCORED_OPEN_PATH.exists(), "can_explain": can_explain,
    })