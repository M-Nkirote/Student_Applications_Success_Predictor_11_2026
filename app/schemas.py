from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

IntakeMonth = Literal["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Undecided"]


class ApplicationInput(BaseModel):
    opportunity_id: Optional[str] = Field(default=None, description="Optional reference only; never used by the model")
    lead_source: str = Field(examples=["Referral"])
    deal_owner: str = Field(examples=["OWNER_3HYMOSBC9T"])
    student_country: str = Field(examples=["Kenya"])
    level_of_study: str = Field(examples=["Postgraduate"])
    destination: str = Field(default="Undecided", examples=["United Kingdom"])
    intake_month: IntakeMonth = "Undecided"
    intake_year: Optional[int] = Field(default=None, ge=2020, le=2035, examples=[2026])
    created_date: date = Field(default_factory=date.today)

    @model_validator(mode="after")
    def intake_needs_a_year(self):
        if self.intake_month != "Undecided" and self.intake_year is None:
            raise ValueError("intake_year is required when intake_month is not 'Undecided'")
        return self


class Contribution(BaseModel):
    feature: str
    label: str
    value: str
    contribution: float


class ScoreResponse(BaseModel):
    opportunity_id: Optional[str] = None
    probability: float
    priority: Literal["High", "Medium", "Low"]
    likely_to_enrol: bool
    average_score: Optional[float] = None
    contributions: Optional[list[Contribution]] = None