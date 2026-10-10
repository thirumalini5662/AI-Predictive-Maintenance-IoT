# AI-Powered Predictive Maintenance for IoT Devices

An interactive Machine Learning project that demonstrates how simulated IoT sensor readings can be used to predict potential machine failure risk.

## Overview

Unexpected equipment failures can cause production delays, maintenance expenses, and operational disruptions. Predictive maintenance uses equipment condition data to identify patterns associated with possible failures.

This project demonstrates a virtual predictive maintenance workflow using Python, a Random Forest classifier, and a Streamlit dashboard. It does not require physical IoT hardware.

## Objectives

- Simulate machine sensor readings.
- Analyze temperature, vibration, and electrical current.
- Train a Machine Learning classification model.
- Predict normal or potential failure-risk conditions.
- Visualize sensor data and model evaluation metrics.
- Explore predictions through an interactive dashboard.

## Features

- Virtual IoT sensor data generation
- Random Forest classification
- Interactive sensor input sliders
- Machine condition and failure-probability display
- Accuracy, precision, recall, and F1-score
- Confusion matrix and classification report
- Sensor distribution charts
- CSV download for dataset and prediction history

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Random Forest Classifier
- Streamlit
- Matplotlib

## System Architecture

Virtual Sensor Data → Data Preparation → Random Forest Model → Failure-Risk Prediction → Dashboard and Alerts

## Sensor Features

| Feature | Description |
|---|---|
| Temperature | Simulated machine temperature in °C |
| Vibration | Simulated vibration level |
| Current | Simulated electrical current in A |
| Failure | Simulated binary target label |

## Installation

1. Install Python and VS Code.
2. Download or clone this repository.
3. Open the project folder in VS Code.
4. Open the terminal and run:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run main.py
```

The dashboard should open in your browser, usually at http://localhost:8501.

## How to Use

1. Launch the Streamlit application.
2. Adjust the temperature, vibration, and current sliders.
3. Click Predict Machine Condition.
4. Review the model prediction and estimated failure probability.
5. Explore the dataset, sensor charts, and evaluation metrics.
6. Download the dataset or prediction history as CSV files.

## Model Evaluation

The model is evaluated using a held-out test split. Metrics include accuracy, precision, recall, F1-score, and a confusion matrix.

The dataset is synthetic, so the results demonstrate the workflow rather than real-world industrial performance.

## Project Limitations

- No physical IoT sensors or live industrial equipment are connected.
- Sensor values and labels are simulated.
- Predictions are demonstrations, not verified maintenance recommendations.
- Prediction history is stored in the current Streamlit session and is not permanent.

## Future Enhancements

- Integrate a documented real-world predictive maintenance dataset.
- Compare multiple Machine Learning algorithms.
- Add time-series analysis and equipment-level history.
- Explore real sensor integration and API-based predictions.

## Learning Outcomes

This project demonstrates practical experience with data preparation, classification, model evaluation, visualization, and interactive Machine Learning applications.

## Author

Malini

GitHub: https://github.com/thirumalini5662

LinkedIn: https://linkedin.com/in/thirumalini-e-70294b299

## Disclaimer

This is an educational prototype using simulated data. It is not designed or validated for industrial safety or real-world maintenance decisions.
