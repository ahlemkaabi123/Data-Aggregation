# Sensor Aggregation & Compression Simulation

This project simulates a wireless sensor network to demonstrate data aggregation and compression techniques for bandwidth optimization. It includes a Python-based simulation core and an interactive Streamlit dashboard.

## Features

- **Synthetic Data Generation**: Simulates temperature and humidity sensors using random walk models.
- **Aggregation**:
    - **Temporal**: Aggregates data over time windows (Mean, Median, Max, Min).
    - **Spatial**: Aggregates data across multiple sensors (Cluster-based).
- **Compression**: Uses `zlib` (LZ77 variant) to compress aggregated payloads.
- **Metrics**: Calculates Compression Ratio, Bandwidth Savings, and Mean Squared Error (MSE).
- **Interactive Dashboard**: Visualizes raw vs. aggregated data and real-time metrics.

## Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.
2.  **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    # Or manually:
    pip install pandas numpy streamlit plotly
    ```

## Usage

### Run the Dashboard
To launch the interactive simulation dashboard:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Run the Test Script
To verify the simulation logic without the UI:

```bash
python test_simulation.py
```

## Project Structure

- `simulation.py`: Core logic for `Sensor` and `Network` classes.
- `processing.py`: Functions for aggregation (`aggregator_temporal`, `aggregator_spatial`) and compression (`compress_data`).
- `app.py`: Streamlit application for the dashboard.
- `test_simulation.py`: Script to verify the logic.

## Technologies Used
- **Python**: Core logic.
- **Streamlit**: Web dashboard.
- **Pandas/Numpy**: Data manipulation and generation.
- **Plotly**: Interactive charts.
- **zlib**: Data compression.
