"""TSAD Orchestra Agent — minimal example.

Sends a time series to the model and asks it to identify anomalies.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are a time series anomaly detection assistant. "
    "When given a numeric time series, identify any anomalous values, "
    "explain why they are anomalous, and suggest the most likely cause."
)


def run(series: list[float]) -> str:
    """Send a time series to the model and return an anomaly analysis.

    Args:
        series: List of numeric sensor readings to analyse.

    Returns:
        The model's explanation of any detected anomalies.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    user_prompt = (
        f"Here is a time series of sensor readings: {series}.\n"
        "Please identify any anomalies, state their indices and values, "
        "and briefly explain what might have caused them."
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("Model returned no text content.")
    return content


if __name__ == "__main__":
    # Mock example: stable signal with two obvious spikes
    mock_series = [10.1, 10.3, 9.9, 10.2, 10.0, 50.0, 10.1, 9.8, 10.4, -30.0, 10.2]
    print(f"Input series: {mock_series}\n")
    answer = run(mock_series)
    print(answer)
