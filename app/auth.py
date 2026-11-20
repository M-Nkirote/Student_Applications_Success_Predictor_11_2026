import os
import secrets
from dotenv import load_dotenv
from fastapi import Request

load_dotenv()

DEMO_EMAIL = os.getenv("AGENT_EMAIL", "amalia@higherstudies.com")
DEMO_PASSWORD = os.getenv("AGENT_PASSWORD", "Dem0123!")


def check_credentials(email: str, password: str) -> bool:
    email_ok = secrets.compare_digest(email.strip().lower(), DEMO_EMAIL.lower())
    password_ok = secrets.compare_digest(password, DEMO_PASSWORD)
    return email_ok and password_ok


def current_agent(request: Request):
    """The logged-in agent's email, or None."""
    return request.session.get("agent")
