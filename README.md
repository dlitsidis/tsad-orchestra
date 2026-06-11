<div align="center">
  <img src="docs/header.png" alt="TSAD-Orchestra Header" />
</div>

**Time Series Anomaly Detection Orchestra** — an intelligent, OpenAI-powered agent system that analyzes numeric time series data and autonomously identifies anomalies using a suite of detection algorithms.

---

## 🌟 Key Features

- **Agentic Anomaly Detection:** Leverages LLMs (via LangChain) to profile time series data and intelligently select the best detection algorithm for the dataset's characteristics.
- **Multiple Detectors Supported:** Includes algorithms from `tsb-uad` such as LOF, HBOS, Isolation Forest, PCA, and Polynomial Fitting.
- **Interactive Streamlit UI:** A rich web interface for selecting datasets, visualizing time series, running detection, and interacting with the agent via chat.
- **TimescaleDB Integration:** Ready to connect to TimescaleDB to seamlessly query and analyze large volumes of time series data.
- **MCP (Model Context Protocol):** Integrates an MCP server via FastMCP, allowing extensible tooling and contextual interactions.
- **CLI & Benchmarking:** Includes a CLI tool for running rapid detection tasks and a benchmarking suite to evaluate model performance.

---

## 🚀 Setup & Installation

### Requirements
- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- (Optional) TimescaleDB database running for data ingestion/analysis

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dlitsidis/tsad-orchestra.git
   cd tsad-orchestra
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.sample .env
   ```
   Open `.env` and set your `OPENAI_API_KEY` and `POSTGRES_*` connection details.

3. **Run the automated setup:**
   The setup script handles installing `uv` (if missing), syncing Python dependencies, starting the required Docker containers, and running database migrations.
   ```bash
   bash scripts/setup.sh
   ```

4. **Activate the virtual environment:**
   ```bash
   # Linux/macOS
   source .venv/bin/activate
   # Windows
   .venv\Scripts\activate
   ```

---

## 💡 Usage

### Running the Streamlit UI
Launch the interactive web application to chat with the agent and visualize anomalies:
```bash
# Using the helper script
./run_ui.sh

# Or directly using Streamlit
streamlit run src/ui.py
```
*The app will automatically open at `http://localhost:8501`.*

### Running via CLI
Run a quick detection task on a specific dataset directly from your terminal:
```bash
uv run python run_detection.py --dataset "your_dataset_id"
```

### Mock Example
Test the system with a built-in mock series containing obvious anomaly spikes:
```bash
uv run python -m src.agent.client
```

---

## 📂 Project Structure

```text
tsad-orchestra/
├── .env.sample               # Environment variables template
├── pyproject.toml            # Project configuration and dependencies
├── README.md                 # Project documentation
├── UI_GUIDE.md               # Detailed guide for the Streamlit UI
├── run_detection.py          # CLI entry point for detection
├── run_ui.sh                 # Helper script to launch the UI
├── benchmark.py              # Benchmarking framework
└── src/                      # Source code
    ├── agent/                # LLM Agent logic, prompts, and models
    ├── benchmark/            # Benchmarking logic and utilities
    ├── utils/                # Helper functions and DB connection tools
    ├── mcp_server.py         # FastMCP Server implementation
    └── ui.py                 # Streamlit UI implementation
```

---

## 🤝 Development & Contributing

Lint and format checks run automatically on every `git commit` via pre-commit hooks.

```bash
# Install the hooks (once, after cloning)
pre-commit install
```

After installation, `ruff check` and `ruff format` run on staged files automatically. To manually format all files, run:
```bash
pre-commit run --all-files
```
