import streamlit as st
import pandas as pd
import joblib

# Load saved model bundle
bundle = joblib.load("girl_model.pkl")
model = bundle["model"]
feature_columns = bundle["feature_columns"]
interaction_pairs = bundle.get("interaction_pairs", [])

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

    rename_map = {
        "Love_Language": "Love Language",
        "Marital_Status": "Marital Status",
    }

    for a, b in interaction_pairs:
        a_name = rename_map.get(a, a)
        b_name = rename_map.get(b, b)

        if a_name not in df.columns or b_name not in df.columns:
            continue

        # Numeric-by-numeric interaction
        if pd.api.types.is_numeric_dtype(df[a_name]) and pd.api.types.is_numeric_dtype(df[b_name]):
            df[f"{a_name}_x_{b_name}"] = df[a_name] * df[b_name]

        # Numeric-by-categorical interaction
        elif pd.api.types.is_numeric_dtype(df[a_name]) and not pd.api.types.is_numeric_dtype(df[b_name]):
            for level in sorted(df[b_name].dropna().unique()):
                df[f"{a_name}_x_{b_name}_{level}"] = (
                    df[a_name] * (df[b_name] == level).astype(int)
                )

        # Categorical-by-numeric interaction
        elif not pd.api.types.is_numeric_dtype(df[a_name]) and pd.api.types.is_numeric_dtype(df[b_name]):
            for level in sorted(df[a_name].dropna().unique()):
                df[f"{a_name}_{b_name}_x_{level}"] = (
                    df[b_name] * (df[a_name] == level).astype(int)
                )

        # Categorical-by-categorical interaction
        else:
            df[f"{a_name}_x_{b_name}"] = (
                df[a_name].astype(str) + "_" + df[b_name].astype(str)
            )

    return df


st.title("Girl A vs Girl B Predictor")
st.write("Enter the person's details below. The app will automatically create the interaction features for prediction.")

gender = st.selectbox("Gender", ["M", "F"])
age = st.number_input("Age", min_value=0, max_value=120, value=30)
love_language = st.selectbox(
    "Love Language",
    ["Acts of Service", "Quality Time"],
)
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
mbti = st.selectbox("MBTI", ["F", "T"])

input_df = pd.DataFrame([{
    "Gender": gender,
    "Age": age,
    "Love Language": love_language,
    "Marital Status": marital_status,
    "MBTI": mbti,
}])

# Make sure the same interaction features are constructed in the app before prediction.
input_df = create_interaction_features(input_df)

# Align to the exact feature order used during training.
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# Predict
after_prediction = model.predict(input_df)[0]

if hasattr(model, "predict_proba"):
    probs = model.predict_proba(input_df)[0]
    prob_df = pd.DataFrame({
        "Girl group": model.classes_,
        "Predicted probability": probs,
    }).sort_values("Predicted probability", ascending=False)

    st.subheader("Prediction")
    st.write(f"Predicted group: {after_prediction}")
    st.dataframe(prob_df.round(4), use_container_width=True)
else:
    st.subheader("Prediction")
    st.write(f"Predicted group: {after_prediction}")