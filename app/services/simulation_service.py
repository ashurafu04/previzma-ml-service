from app.schemas import SimulationRequest, SimulationResponse

MODEL_VERSION = "mock-simulation-v1"


def run_simulation(request: SimulationRequest) -> SimulationResponse:
    if not request.sales_history:
        result_value = 0.0
    else:
        baseline = sum(item.amount for item in request.sales_history) / len(
            request.sales_history
        )
        result_value = baseline * (1 + request.input_change_percent / 100)

    return SimulationResponse(
        result_value=round(max(result_value, 0.0), 2),
        model_version=MODEL_VERSION,
    )
