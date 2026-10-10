
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)


# --------------------------------------------------
# 1. GENERATE SIMULATED IOT SENSOR DATA
# --------------------------------------------------
@st.cache_data
def make_dataset(seed=42, n=3000):
    rng = np.random.default_rng(seed)

    temperature = rng.normal(52, 12, n).clip(20, 105)
    vibration = rng.gamma(2.0, 1.0, n).clip(0, 10)
    current = rng.normal(7.0, 2.0, n).clip(0.5, 16)

    risk_score = (
        0.055 * (temperature - 55)
        + 0.62 * (vibration - 2.0)
        + 0.23 * (current - 7.0)
        + rng.normal(0, 1.15, n)
    )

    probability = 1 / (1 + np.exp(-risk_score))
    failure = rng.binomial(1, probability)

    return pd.DataFrame({
        "temperature_c": temperature.round(2),
        "vibration_level": vibration.round(3),
        "current_a": current.round(2),
        "failure": failure.astype(int),
    })


# --------------------------------------------------
# 2. TRAIN THE MACHINE LEARNING MODEL
# --------------------------------------------------
@st.cache_resource
def train_model(seed=42):
    df = make_dataset(seed=seed)

    features = ["temperature_c", "vibration_level", "current_a"]
    target = "failure"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=160,
                max_depth=10,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
    }

    matrix = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    report = classification_report(
        y_test, predictions, zero_division=0
    )

    return df, model, metrics, matrix, report


# --------------------------------------------------
# 3. APPLICATION HEADER
# --------------------------------------------------
st.title("⚙️ AI-Powered Predictive Maintenance for IoT Devices")
st.caption(
    "IoT Sensor Simulation | Random Forest | Predictive Analytics"
)

st.warning(
    "Academic demonstration only. This application uses simulated "
    "sensor data, not live industrial equipment. Predictions must "
    "not be used as a real-world machine safety system."
)


# --------------------------------------------------
# 4. SIDEBAR
# --------------------------------------------------
if "dataset_seed" not in st.session_state:
    st.session_state.dataset_seed = 42

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

with st.sidebar:
    st.header("Project Controls")

    st.write("Dataset: 3,000 simulated sensor readings")

    if st.button("🔄 Regenerate Dataset", use_container_width=True):
        st.session_state.dataset_seed += 1
        st.rerun()

    if st.button("🧠 Retrain Model", use_container_width=True):
        train_model.clear()
        st.rerun()

    st.divider()
    st.write("**Input sensor features**")
    st.caption("Temperature, vibration and electrical current")


# --------------------------------------------------
# 5. LOAD DATA AND MODEL
# --------------------------------------------------
with st.spinner("Preparing sensor data and training AI model..."):
    df, model, metrics, cm, report = train_model(
        st.session_state.dataset_seed
    )

features = ["temperature_c", "vibration_level", "current_a"]


# --------------------------------------------------
# 6. KEY PERFORMANCE INDICATORS
# --------------------------------------------------
st.subheader("Model Performance Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Dataset Rows", f"{len(df):,}")
c2.metric("Failure Examples", f"{int(df['failure'].sum()):,}")
c3.metric("Model Accuracy", f"{metrics['accuracy']:.1%}")
c4.metric("Failure Recall", f"{metrics['recall']:.1%}")


# --------------------------------------------------
# 7. SENSOR INPUTS
# --------------------------------------------------
st.subheader("1. Simulate IoT Sensor Readings")

left, right = st.columns(2)

with left:
    temperature = st.slider(
        "Temperature (°C)", 20.0, 105.0, 72.0, 0.5
    )

    vibration = st.slider(
        "Vibration Level", 0.0, 10.0, 4.5, 0.1
    )

    current = st.slider(
        "Electrical Current (A)", 0.5, 16.0, 10.5, 0.1
    )

    predict_clicked = st.button(
        "🔍 Predict Machine Condition",
        type="primary",
        use_container_width=True,
    )

with right:
    st.info(
        "Adjust the sliders to simulate a machine's sensor readings. "
        "Actual safe operating limits depend on the equipment."
    )

    st.write("**Current sensor readings**")

    st.dataframe(
        pd.DataFrame({
            "Sensor": [
                "Temperature (°C)",
                "Vibration Level",
                "Current (A)",
            ],
            "Reading": [temperature, vibration, current],
        }),
        hide_index=True,
        use_container_width=True,
    )


# --------------------------------------------------
# 8. PREDICT MACHINE CONDITION
# --------------------------------------------------
input_data = pd.DataFrame([{
    "temperature_c": temperature,
    "vibration_level": vibration,
    "current_a": current,
}])

prediction = int(model.predict(input_data[features])[0])
failure_probability = float(
    model.predict_proba(input_data[features])[0][1]
)

st.subheader("2. AI Prediction")

p1, p2 = st.columns([1, 2])

with p1:
    if prediction == 1:
        st.error("⚠️ FAILURE RISK PREDICTED")
    else:
        st.success("✅ NORMAL CONDITION PREDICTED")

    st.metric(
        "Estimated Failure Probability",
        f"{failure_probability:.1%}",
    )

with p2:
    st.write("**Predicted failure probability**")
    st.progress(
        min(max(failure_probability, 0.0), 1.0)
    )

    if prediction == 1:
        st.warning(
            "Suggested action: inspect the equipment and verify "
            "the readings with qualified personnel."
        )
    else:
        st.info(
            "Suggested action: continue routine monitoring and "
            "follow the equipment maintenance plan."
        )

if predict_clicked:
    st.session_state.prediction_history.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Temperature (°C)": temperature,
        "Vibration Level": vibration,
        "Current (A)": current,
        "Prediction": (
            "Failure Risk" if prediction == 1 else "Normal"
        ),
        "Failure Probability": round(failure_probability, 4),
    })

    st.success("Prediction added to the session history.")


# --------------------------------------------------
# 9. DATASET AND VISUALIZATIONS
# --------------------------------------------------
st.subheader("3. Sensor Dataset and Analytics")

tab1, tab2, tab3, tab4 = st.tabs([
    "Dataset",
    "Sensor Charts",
    "Model Evaluation",
    "Prediction History",
])

with tab1:
    st.write("Sample sensor records")
    st.dataframe(df.head(50), use_container_width=True)

    st.download_button(
        "⬇️ Download Sensor Dataset CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="simulated_iot_sensor_data.csv",
        mime="text/csv",
    )

with tab2:
    chart1, chart2 = st.columns(2)

    with chart1:
        fig, ax = plt.subplots()
        df["temperature_c"].hist(ax=ax, bins=25)
        ax.set_title("Temperature Distribution")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)

    with chart2:
        fig, ax = plt.subplots()
        df["vibration_level"].hist(ax=ax, bins=25)
        ax.set_title("Vibration Distribution")
        ax.set_xlabel("Vibration Level")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)

    fig, ax = plt.subplots()
    df.boxplot(column="current_a", by="failure", ax=ax)
    ax.set_title("Current by Simulated Failure Label")
    ax.set_xlabel("Failure Label (0 = Normal, 1 = Failure)")
    ax.set_ylabel("Current (A)")
    fig.suptitle("")
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    st.write("**Held-out test dataset performance**")

    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    b.metric("Precision", f"{metrics['precision']:.1%}")
    c.metric("Recall", f"{metrics['recall']:.1%}")
    d.metric("F1-score", f"{metrics['f1_score']:.1%}")

    fig, ax = plt.subplots()
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Failure"],
    )
    display.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    plt.close(fig)

    st.write("**Classification Report**")
    st.code(report)

    st.caption(
        "These metrics describe performance on simulated data. "
        "They do not prove real-world industrial accuracy."
    )

with tab4:
    history = st.session_state.prediction_history

    if history:
        history_df = pd.DataFrame(history)
        st.dataframe(
            history_df.iloc[::-1],
            use_container_width=True,
        )

        st.download_button(
            "⬇️ Download Prediction History CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
        )

        if st.button("Clear Prediction History"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info(
            "No predictions saved in this session yet. "
            "Click Predict Machine Condition to add one."
        )


# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "Built with Python, Streamlit, Pandas, NumPy, "
    "Scikit-learn and Matplotlib."
)
