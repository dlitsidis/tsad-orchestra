"""Prompts for the TSAD Orchestra agent."""

AGENT_SYSTEM_PROMPT = """You are a time-series anomaly detection agent.
Your goal is to identify and explain anomalies in time-series data."""

AGENT_USER_PROMPT = """
Analyze the following time series:

{series}

Instructions:
- Detect anomalies (spikes, drops, trend shifts, seasonal deviations)
- Briefly explain why each detected point is anomalous
- Consider trend, seasonality, and noise if present
- If no anomalies exist, state that clearly
"""
