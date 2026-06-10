# TSAD Orchestra — Streamlit UI

Web-based interface for the TSAD Orchestra anomaly detection agent.

## Features

- 🎯 **Time Series Selection**: Browse and select any time series from your TimescaleDB database
- 💬 **Chat Interface**: Interact with the anomaly detection agent
- 📊 **Interactive Visualization**: View time series data with detected anomalies highlighted
- 📈 **Statistical Summary**: Quick view of key statistics for the selected series
- 🔍 **One-Click Detection**: Run automated anomaly detection analysis
- 📋 **Detailed Results**: View complete anomaly reports with indices and values

## Prerequisites

Before running the UI, ensure you have:

1. **Python 3.11+** installed
2. **TSAD Orchestra project** cloned and dependencies installed (see main README.md)
3. **TimescaleDB** running with time series data loaded
4. **Environment variables configured** in `.env` file:
   ```
   OPENAI_API_KEY=<your_openai_api_key>
   POSTGRES_USER=<db_user>
   POSTGRES_PASSWORD=<db_password>
   POSTGRES_HOST=<db_host>
   POSTGRES_PORT=5432
   POSTGRES_DB=<db_name>
   ```

## Installation

1. **Install dependencies** (if not already done):
   ```bash
   uv sync
   ```

2. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Streamlit and Plotly** (now included in dependencies):
   ```bash
   uv sync  # This will install all dependencies including streamlit and plotly
   ```

## Running the UI

### Option 1: Using the provided script (Linux/macOS)

```bash
chmod +x run_ui.sh
./run_ui.sh
```

### Option 2: Direct Streamlit command

```bash
streamlit run src/ui.py
```

The app will open in your default browser at `http://localhost:8501`

### Option 3: Running with custom port

```bash
streamlit run src/ui.py --server.port 8502
```

## Usage Guide

### 1. **Selecting a Time Series**

- Use the "Select a Time Series" dropdown menu on the left sidebar
- Choose from all available tables in your database
- The series statistics will automatically update

### 2. **Viewing Statistics**

Once a series is selected, the sidebar displays:
- **Data Points**: Number of observations in the series
- **Mean**: Average value
- **Std Dev**: Standard deviation
- **Min/Max**: Minimum and maximum values
- **Range**: Difference between max and min

### 3. **Running Anomaly Detection**

There are two ways to start the analysis:

**Method A: Quick Detection Button**
- Click the blue "🔍 Run Detection" button
- The agent analyzes the series and reports findings

**Method B: Chat Input**
- Type a message like "analyze this time series" or "detect anomalies"
- Press Enter or send the message
- The agent will respond and run detection if requested

### 4. **Viewing Results**

After detection completes:

- **Interactive Chart**: The time series is displayed with anomalies marked in red (×)
  - Hover over points to see exact values
  - Use zoom/pan controls to explore
  
- **Detection Summary**: Shows the total number of anomalies found
  
- **Expandable Sections**:
  - **View Summary**: Explanation of detector selection and findings
  - **View All Anomalies**: Detailed table of detected anomalies with indices and values

### 5. **Chat History**

All interactions are preserved in the chat history, allowing you to:
- Review previous analyses
- Track which detectors were used
- Compare results across different time series (by changing selection and re-running)

## How It Works

1. **Series Selection**: Select a time series from the database via the sidebar
2. **Agent Invocation**: On "Run Detection" or chat request:
   - Receives time series data
   - Profiles data characteristics
   - Selects an anomaly detection algorithm
   - Executes the detector
   - Reports results
3. **Visualization**: Detected anomalies are highlighted on the interactive chart
4. **Results Storage**: Results are stored in Streamlit session state

## Available Anomaly Detection Algorithms

The agent automatically selects from:

- **LOF (Local Outlier Factor)**: For contextual/local anomalies
- **HBOS (Histogram-based Outlier Score)**: For fast global outlier detection
- **Isolation Forest**: General-purpose anomaly detection
- **PCA (Principal Component Analysis)**: For structural anomalies
- **Polynomial Fitting**: For trend-based anomalies

## Troubleshooting

### "No time series found in the database"

- Ensure your TimescaleDB is running and accessible
- Check that time series data has been loaded into the database
- Verify POSTGRES_* environment variables in .env

### "Missing OPENAI_API_KEY"

- Ensure `OPENAI_API_KEY` is set in your `.env` file
- The key should be a valid OpenAI API key

### "Error loading time series"

- Check the database connection
- Ensure the selected time series table exists and contains data
- Review database logs for more details

### "Detection takes too long"

- Larger time series may take longer to analyze
- This is normal, as the agent profiles the data and runs detectors
- Anomaly detection on series with 10,000+ points may take 1-2 minutes

### "App resets when clicking detection"

- This is normal Streamlit behavior (page rerun)
- Results are preserved in the chat and visualization

## Configuration

You can customize the Streamlit app behavior by editing `.streamlit/config.toml`:

- **Theme colors**: Change `primaryColor`, `backgroundColor`, etc.
- **Layout**: Adjust `hideSidebarNav`, `hideTopBar`
- **Server settings**: Modify `runOnSave`, `maxUploadSize`

## Development

To modify the UI, edit `src/ui.py`:

- **Main function**: `main()` - Entry point
- **Series loading**: `load_available_series()`, `load_time_series_data()`
- **Detection**: `run_anomaly_detection_sync()`
- **Visualization**: `plot_time_series()`

## Performance Tips

- **Cache frequently accessed data**: Streamlit automatically caches some operations
- **Use smaller time series initially**: Start with series < 5000 points
- **Monitor API usage**: OpenAI API calls are made during detection

## Next Steps

- Try analyzing different time series with various characteristics
- Experiment with different types of anomalies
- Integrate results with downstream systems
- Customize the agent's behavior in `src/agent/prompts.py`

## Support

For issues or questions:
1. Check this guide's Troubleshooting section
2. Review the main README.md in the project root
3. Check error logs (visible in the Streamlit sidebar or terminal)
4. Ensure all environment variables are correctly configured
