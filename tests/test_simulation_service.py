from app.schemas import SimulationRequest
from app.services.simulation_service import run_simulation


def simulation_request(
    scenario_type: str,
    input_change_percent: float,
    baseline_value: float = 1000.0,
) -> SimulationRequest:
    return SimulationRequest(
        company_id=1,
        user_id=7,
        product_id=10,
        product_name="Industrial Pump",
        product_sku="PUMP-001",
        client_segment_id=20,
        client_segment_name="Grand compte",
        client_segment_type="GRAND_COMPTE",
        scenario_type=scenario_type,
        input_change_percent=input_change_percent,
        baseline_value=baseline_value,
        sales_history=[],
    )


def test_simulation_price_change_applies_percent_to_baseline_value() -> None:
    response = run_simulation(simulation_request("PRICE_CHANGE", 10.0))

    assert response.result_value == 1100.0
    assert response.model_version == "baseline-simulation-v1"


def test_simulation_demand_change_applies_percent_to_baseline_value() -> None:
    response = run_simulation(simulation_request("DEMAND_CHANGE", 15.0))

    assert response.result_value == 1150.0


def test_simulation_supply_delay_reduces_baseline_value() -> None:
    response = run_simulation(simulation_request("SUPPLY_DELAY", 20.0))

    assert response.result_value == 800.0


def test_simulation_discount_campaign_balances_price_drop_and_demand_uplift() -> None:
    response = run_simulation(simulation_request("DISCOUNT_CAMPAIGN", 10.0))

    assert response.result_value == 945.0
