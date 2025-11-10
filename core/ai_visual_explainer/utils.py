def format_valuations(valuations: dict) -> str:
    return "\n".join([f"- {model}: ${value:,.2f}" for model, value in valuations.items()])
