import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model bundle
bundle = joblib.load("girl_model.pkl")
model = bundle["model"]
feature_columns = bundle["feature_columns"]
interaction_pairs = bundle.get("interaction_pairs", [])
numeric_features = bundle.get("numeric_features", ["Age"])
categorical_features = bundle.get(
    "categorical_features",
    ["Gender", "Love Language", "Marital Status", "MBTI"],
)

# Default interaction pairs if the model bundle was saved before interaction metadata was added.
if not interaction_pairs:
    interaction_pairs = [
        ("Gender", "Love Language"),
        ("Gender", "Marital Status"),
        ("Age", "Gender"),
    ]


def create_interaction_features(df):
    """Add the same interaction features used when the model was trained."""
    df = df.copy()

    for a, b in interaction_pairs:
        if a not in df.columns or b not in df.columns:
            continue

        if pd.api.types.is_numeric_dtype(df[a]) and pd.api.types.is_numeric_dtype(df[b]):
            df[f"{a}_x_{b}"] = df[a] * df[b]

        elif pd.api.types.is_numeric_dtype(df[a]) and not pd.api.types.is_numeric_dtype(df[b]):
            for level in sorted(df[b].dropna().unique()):
                df[f"{a}_x_{b}_{level}"] = (
                    df[a] * (df[b] == level).astype(int)
                )

        elif not pd.api.types.is_numeric_dtype(df[a]) and pd.api.types.is_numeric_dtype(df[b]):
            for level in sorted(df[a].dropna().unique()):
                df[f"{a}_{b}_x_{level}"] = (
                    df[b] * (df[a] == level).astype(int)
                )

        else:
            df[f"{a}_x_{b}"] = (
                df[a].astype(str) + "_" + df[b].astype(str)
            )

    return df


def align_to_training_columns(df, expected_columns):
    """Align columns to the training order and replace missing values with safe values."""
    df = df.copy()

    for col in expected_columns:
        if col not in df.columns:
            if col in numeric_features:
                df[col] = np.nan
            else:
                df[col] = None

    df = df[expected_columns].copy()

    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype("object")
            df[col] = df[col].where(df[col].notna(), None)

    return df


st.title("Girl A vs Girl B Predictor")
st.write("Enter the person's details below. The app will automatically create the interaction terms before prediction.")

gender = st.selectbox("Gender", ["M", "F"])
age = st.number_input("Age", min_value=0, max_value=120, value=30)
love_language = st.selectbox(
    "Love Language",
    ["Acts of Service", "Quality Time", "Words of Affirmation", "Physical Touch", "Receiving Gifts"],
)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
mbti = st.selectbox("MBTI", ["F", "T"])

raw_df = pd.DataFrame([{
    "Gender": gender,
    "Age": age,
    "Love Language": love_language,
    "Marital Status": marital_status,
    "MBTI": mbti,
}])

# Build the interaction features exactly like in the notebook.
input_df = create_interaction_features(raw_df)

# Final alignment to the variable order used during training.
input_df = align_to_training_columns(input_df, feature_columns)

# Predict
prediction = model.predict(input_df)[0]

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(input_df)[0]
    prob_df = pd.DataFrame({
        "Girl group": model.classes_,
        "Predicted probability": probabilities,
    }).sort_values("Predicted probability", ascending=False)

    st.subheader("Prediction")
    st.write(f"Predicted group: {prediction}")
    st.dataframe(prob_df.round(4), use_container_width=True)
else:
    st.subheader("Prediction")
    st.write(f"Predicted group: {prediction}")