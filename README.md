# tsad-orchestra

**Time Series Anomaly Detection Orchestra** — an OpenAI-powered agent that analyses numeric time series and identifies anomalies.


---

## Requirements

- Python 3.11+
- An [OpenAI API key]

---

## Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/dlitsidis/tsad-orchestra.git
cd tsad-orchestra

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create a virtual environment and install the project
uv sync

# 4. Activate the virtual environment
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 5. Configure environment variables
cp .env.sample .env
# Open .env and set OPENAI_API_KEY=<your key>
```

---


## Pre-commit hooks

Lint and format checks run automatically on every `git commit`:

```bash
# Install the hooks (once, after cloning)
pre-commit install
```

After that, `ruff check` and `ruff format` run on staged files before each commit. To run manually against all files:

```bash
pre-commit run --all-files
```

---

## Run the mock example

`client.py` ships with a built-in mock series that has two obvious spikes
(`50.0` at index 5 and `-30.0` at index 9):

```bash
uv run python -m src.agent.client
```

