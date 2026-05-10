from app.schemas import SimulationRequest, SimulationResponse
from app.services.baseline import calculate_revenue_baseline

MODEL_VERSION = "baseline-simulation-v1"

PRICE_CHANGE = "PRICE_CHANGE"
DEMAND_CHANGE = "DEMAND_CHANGE"
SUPPLY_DELAY = "SUPPLY_DELAY"
DISCOUNT_CAMPAIGN = "DISCOUNT_CAMPAIGN"


def run_simulation(request: SimulationRequest) -> SimulationResponse:
    baseline_value = (
        request.baseline_value
        if request.baseline_value is not None
        else calculate_revenue_baseline(request.sales_history).monthly_value
    )

    result_value = baseline_value * _scenario_multiplier(
        request.scenario_type,
        request.input_change_percent,
    )
    result_value = max(result_value, 0.0)
    impact_value = result_value - baseline_value
    impact_percent = (
        impact_value / baseline_value * 100 if baseline_value > 0 else 0.0
    )

    return SimulationResponse(
        baseline_value=round(baseline_value, 2),
        result_value=round(result_value, 2),
        impact_value=round(impact_value, 2),
        impact_percent=round(impact_percent, 4),
        model_version=MODEL_VERSION,
    )


def _scenario_multiplier(scenario_type: str, input_change_percent: float) -> float:
    change_rate = input_change_percent / 100

    if scenario_type == PRICE_CHANGE:
        return 1 + change_rate

    if scenario_type == DEMAND_CHANGE:
        return 1 + change_rate

    if scenario_type == SUPPLY_DELAY:
        return max(0.0, 1 - change_rate)

    if scenario_type == DISCOUNT_CAMPAIGN:
        price_multiplier = max(0.0, 1 - change_rate)
        demand_uplift_multiplier = max(0.0, 1 + change_rate * 0.5)
        return price_multiplier * demand_uplift_multiplier

    return 1.0
