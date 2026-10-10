# AI-Powered Predictive Maintenance for IoT Devices

A beginner-friendly Machine Learning project that demonstrates how simulated IoT sensor readings can be used to classify machine condition as **normal** or **failure risk**. It includes a Streamlit dashboard, interactive sensor controls, model evaluation, charts, and CSV exports.

> **Important limitation:** This is an academic prototype using synthetic, simulated data. It is not connected to physical IoT sensors and must not be used for industrial safety or maintenance decisions. Results on synthetic data do not demonstrate real-world model performance.

## Problem statement
Unexpected machine failures can cause downtime, repair costs, and production delays. Predictive maintenance uses condition data to identify patterns associated with possible equipment problems. This project demonstrates the data and ML workflow in a laptop-only simulation.

## Features
- Generate a simulated dataset with temperature, vibration, current, and a failure label
- Train a Random Forest classifier
- Evaluate with accuracy, precision, recall, F1-score, and confusion matrix
- Adjust simulated sensor readings and predict machine condition
- Display sensor distributions and a dataset preview
- Save prediction history and download CSV files
- Save the trained model and evaluation metrics locally

## Tech stack
Python, Pandas, NumPy, Scikit-learn, Streamlit, Matplotlib, Joblib.

## Project structure
```text
AI-Predictive-Maintenance-IoT/
├── data/
│   └── simulated_iot_sensor_data.csv  # created automatically
├── models/
│   └── predictive_maintenance_model.joblib  # created automatically
├── outputs/
│   ├── model_metrics.json  # created automatically
│   └── prediction_history.csv  # created after predictions
├── images/                 # add screenshots here
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run on Windows with VS Code
1. Install Python 3.11 or 3.12 and Git.
2. Open this project folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment:
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use Command Prompt and run `.venv\Scripts\activate.bat`.
5. Install packages:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
6. Start the dashboard:
   ```powershell
   streamlit run main.py
   ```
7. A browser tab should open with the dashboard. If not, open the local URL printed in the terminal (usually `http://localhost:8501`).

## How to use
- Adjust temperature, vibration, and current using the sliders.
- Click **Predict machine condition**.
- Inspect the predicted class and model-estimated probability.
- Explore the Dataset, Sensor distributions, and Model evaluation tabs.
- Download the simulated dataset or prediction history.
- Use the sidebar to regenerate data or retrain the model.

## Model and evaluation
A Random Forest classifier is trained on an 80/20 stratified train/test split. Reported metrics are calculated on the held-out portion of the simulated dataset. Because the labels are generated from a synthetic risk process, these metrics are for demonstrating the workflow only and should not be represented as accuracy on real industrial equipment.

## Future improvements
- Replace simulated data with a documented real predictive-maintenance dataset
- Add chronological validation for time-series sensor data
- Add equipment IDs, timestamps, and maintenance event history
- Compare multiple algorithms and tune thresholds for failure recall
- Add a real sensor/API integration only after safe hardware and deployment design

## Suggested GitHub topics
`predictive-maintenance`, `machine-learning`, `iot`, `python`, `scikit-learn`, `streamlit`, `industrial-analytics`

## Author
Add your name, course, and contact links here.
