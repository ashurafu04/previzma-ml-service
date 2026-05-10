from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def to_camel(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class HealthResponse(CamelModel):
    status: str


class SalesHistoryItem(CamelModel):
    sale_date: date
    quantity: int = Field(..., ge=0)
    amount: float = Field(..., ge=0)
    confirmed_order: bool
    source_status: str = Field(..., min_length=1)


class ForecastRequest(CamelModel):
    company_id: int = Field(..., ge=1)
    product_id: int = Field(..., ge=1)
    product_name: str = Field(..., min_length=1)
    product_sku: str = Field(..., min_length=1)
    client_segment_id: int = Field(..., ge=1)
    client_segment_name: str = Field(..., min_length=1)
    client_segment_type: str = Field(..., min_length=1)
    horizon: int = Field(..., gt=0)
    calculation_date: date
    sales_history: list[SalesHistoryItem] = Field(default_factory=list)


class ForecastResponse(CamelModel):
    predicted_value: float = Field(..., ge=0)
    confidence_score: float | None = Field(default=None, ge=0, le=1)
    model_version: str = Field(..., min_length=1)
    calculation_date: date


class SimulationRequest(CamelModel):
    company_id: int = Field(..., ge=1)
    user_id: int = Field(..., ge=1)
    product_id: int = Field(..., ge=1)
    product_name: str = Field(..., min_length=1)
    product_sku: str = Field(..., min_length=1)
    client_segment_id: int = Field(..., ge=1)
    client_segment_name: str = Field(..., min_length=1)
    client_segment_type: str = Field(..., min_length=1)
    scenario_type: Literal[
        "PRICE_CHANGE",
        "DEMAND_CHANGE",
        "SUPPLY_DELAY",
        "DISCOUNT_CAMPAIGN",
    ]
    input_change_percent: float = Field(..., ge=-100)
    baseline_value: float | None = Field(default=None, ge=0)
    sales_history: list[SalesHistoryItem] = Field(default_factory=list)


class SimulationResponse(CamelModel):
    result_value: float = Field(..., ge=0)
    model_version: str = Field(..., min_length=1)
