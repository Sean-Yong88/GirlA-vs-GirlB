import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

print("CURRENT WORKING DIRECTORY:", os.getcwd())
print("MODEL PATH:", os.path.abspath("girl_model.pkl"))
print("MODEL EXISTS:", os.path.exists("girl_model.pkl"))

bundle = joblib.load("girl_model.pkl")
print("MODEL CLASSES:", bundle["model"].classes_)
print("FEATURE COLUMNS:", bundle["feature_columns"])
print("BUNDLE TYPE:", type(bundle["model"]).__name__)

model = bundle["model"]
feature_columns = bundle["feature_columns"]
numeric_features = bundle.get("numeric_features", [])
categorical_features = bundle.get("categorical_features", [])

def build_exact_interactions(df):
    df = df.copy()

    # Exact interaction columns used in training
    df["Gender_x_Love Language"] = (
        df["Gender"].astype(str) + "_" + df["Love Language"].astype(str)
    )

    for level in ["Acts of Service", "Quality Time"]:
        df[f"Age_x_Love Language_{level}"] = (
            df["Age"] * (df["Love Language"].astype(str) == level).astype(int)
        )

    for level in ["Married", "Single"]:
        df[f"Age_x_Marital Status_{level}"] = (
            df["Age"] * (df["Marital Status"].astype(str) == level).astype(int)
        )

    return df


def prepare_input(df):
    df = df.copy()

    required_cols = ["Gender", "Age", "Love Language", "Marital Status", "MBTI"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Clean strings
    for col in ["Gender", "Love Language", "Marital Status", "MBTI"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("Missing")
                .astype(str)
                .replace({"nan": "Missing", "None": "Missing", "<NA>": "Missing"})
            )

    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    df = build_exact_interactions(df)

    for col in feature_columns:
        if col not in df.columns:
            if col in numeric_features:
                df[col] = np.nan
            else:
                df[col] = "Missing"

    for col in categorical_features:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("Missing")
                .astype(str)
                .replace({"nan": "Missing", "None": "Missing", "<NA>": "Missing"})
            )

    df = df[feature_columns]
    return df


st.title("Girl A vs Girl B Predictor")
st.write("Enter the person's details below. The app will automatically build the same interaction features used in training.")

gender = st.selectbox("Gender", ["M", "F"])
age = st.number_input("Age", min_value=0, max_value=120, value=30)
love_language = st.selectbox("Love Language", ["Acts of Service", "Quality Time"])
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
mbti = st.selectbox("MBTI", ["F", "T"])

raw_input = pd.DataFrame([{
    "Gender": gender,
    "Age": age,
    "Love Language": love_language,
    "Marital Status": marital_status,
    "MBTI": mbti,
}])

input_df = prepare_input(raw_input)

print("FINAL INPUT ROW:")
print(input_df)
print("FINAL INPUT DTYPE:")
print(input_df.dtypes)

prediction = model.predict(input_df)[0]
print("APP PREDICTION:", prediction)

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(input_df)[0]
    print("APP PROBABILITIES:", probabilities)
    prob_df = pd.DataFrame({
        "Girl group": model.classes_,
        "Predicted probability": probabilities
    }).sort_values("Predicted probability", ascending=False)

    st.subheader("Prediction")
    st.write(f"Predicted group: {prediction}")
    st.dataframe(prob_df.round(4), use_container_width=True)
else:
    st.subheader("Prediction")
    st.write(f"Predicted group: {prediction}")