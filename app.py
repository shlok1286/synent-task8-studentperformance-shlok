import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Paths
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "StudentsPerformance.csv"
MODEL_PATH = ROOT / "model.pkl"


def load_data():
    return pd.read_csv(DATA_PATH)


def preprocess(df):
    df_proc = pd.get_dummies(df, drop_first=True)
    y = df_proc['math score']
    X = df_proc.drop('math score', axis=1)
    return X, y


def train_and_save_model():
    df = load_data()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    payload = {"model": model, "columns": X.columns.tolist()}
    joblib.dump(payload, MODEL_PATH)
    return payload


def load_or_train_model():
    if MODEL_PATH.exists():
        try:
            payload = joblib.load(MODEL_PATH)
            # basic integrity check
            if 'model' in payload and 'columns' in payload:
                return payload
        except Exception:
            pass
    return train_and_save_model()


def predict(payload, input_dict):
    model = payload['model']
    columns = payload['columns']
    # Build raw input DF with original columns
    raw = pd.DataFrame([input_dict])
    raw_proc = pd.get_dummies(raw, drop_first=True)
    # Align with training columns
    # Create a single-row DataFrame aligned to training columns
    row = {c: (raw_proc.iloc[0][c] if c in raw_proc.columns else 0) for c in columns}
    X_in = pd.DataFrame([row], columns=columns).fillna(0).astype(float)
    pred = model.predict(X_in)[0]
    return float(pred)


def main():
    st.set_page_config(page_title="Student Performance Predictor", layout="wide")
    st.title("Student Performance Prediction — Math Score")

    st.markdown(
        "This interactive app reuses the project's original preprocessing and Linear Regression model to predict a student's math score from demographic and test scores."
    )

    # Left: controls
    with st.sidebar:
        st.header("Controls")
        show_data = st.checkbox("Show dataset preview", value=False)
        show_images = st.checkbox("Show saved visuals", value=True)
        retrain = st.button("Retrain model now")

    # Load dataset and model
    df = load_data()
    payload = load_or_train_model()

    if retrain:
        st.info("Retraining model — this may take a few seconds.")
        payload = train_and_save_model()
        st.success("Retrained and saved model to model.pkl")

    # Top metrics
    st.subheader("Dataset overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Features", df.shape[1] - 1)
    col3.metric("Target", "math score")

    if show_data:
        st.dataframe(df.head(100))

    if show_images:
        st.subheader("Project Visuals")
        st.image(str(ROOT / "Images" / "ActualVsPredictedScores.png"), caption="Actual vs Predicted")
        st.image(str(ROOT / "Images" / "LinearRegressionResults.png"), caption="Linear Regression Results")

    st.markdown("---")

    st.subheader("Predict math score for a student")
    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        gender = c1.selectbox("Gender", options=['female', 'male'])
        race = c1.selectbox("Race / Ethnicity", options=[
            'group A', 'group B', 'group C', 'group D', 'group E'
        ])
        parent_edu = c1.selectbox("Parental level of education", options=[
            "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"
        ])
        lunch = c2.selectbox("Lunch", options=['standard', 'free/reduced'])
        test_prep = c2.selectbox("Test preparation course", options=['none', 'completed'])
        reading = st.slider("Reading score", 0, 100, 70)
        writing = st.slider("Writing score", 0, 100, 70)
        submitted = st.form_submit_button("Predict")

    if submitted:
        # Build a raw input dict matching original dataset column names
        input_dict = {
            'gender': gender,
            'race/ethnicity': race,
            'parental level of education': parent_edu,
            'lunch': lunch,
            'test preparation course': test_prep,
            'reading score': reading,
            'writing score': writing,
            'math score': 0  # placeholder — removed by preprocess when training
        }
        pred = predict(payload, input_dict)
        st.success(f"Predicted math score: {pred:.2f} (out of 100)")

    st.markdown("---")
    st.subheader("Notes")
    st.markdown(
        "- The model is a Linear Regression trained on the repository dataset.\n- You can retrain the model with the sidebar button; retraining overwrites `model.pkl`."
    )


if __name__ == "__main__":
    main()
