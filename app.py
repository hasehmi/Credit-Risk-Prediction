"""
Credit Risk Prediction — Streamlit App
Loads the trained XGBoost model and explains each prediction with SHAP.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import re
import os

st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="centered")

# Load artifacts from the same directory as this script, so the app works
# no matter where it's run from (not just when the terminal's cwd happens
# to be this folder).
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_PATH, "xgb_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_PATH, "scaler.pkl"))
    model_columns = joblib.load(os.path.join(BASE_PATH, "model_columns.pkl"))
    best_threshold = joblib.load(os.path.join(BASE_PATH, "best_threshold.pkl"))
    numeric_cols = joblib.load(os.path.join(BASE_PATH, "numeric_cols.pkl"))
    return model, scaler, model_columns, best_threshold, numeric_cols

model, scaler, model_columns, best_threshold, numeric_cols = load_artifacts()
explainer = shap.TreeExplainer(model)

def clean_col(c):
    c = re.sub(r"[\[\]<]", "", c)
    c = re.sub(r"\s+", "_", c)
    c = re.sub(r"/", "_", c)
    return c

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("💳 Credit Risk Predictor")
st.markdown(
    "Estimates the probability that a loan applicant will default, "
    "based on the [German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) (UCI). "
    "Every prediction includes a SHAP breakdown of *why* the model scored it that way."
)

with st.form("applicant_form"):
    st.subheader("Applicant details")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 80, 35)
        duration_months = st.slider("Loan duration (months)", 4, 72, 24)
        credit_amount = st.number_input("Credit amount (DM)", min_value=250, max_value=20000, value=3000, step=100)
        installment_rate_pct = st.slider("Installment rate (% of disposable income)", 1, 4, 2)
        present_residence_since = st.slider("Years at present residence", 1, 4, 2)
        existing_credits_count = st.slider("Existing credits at this bank", 1, 4, 1)
        num_dependents = st.slider("Number of dependents", 1, 2, 1)

    with col2:
        checking_account_status = st.selectbox(
            "Checking account status",
            ["< 0 DM", "0-200 DM", ">= 200 DM", "no checking account"]
        )
        savings_account = st.selectbox(
            "Savings account balance",
            ["< 100 DM", "100-500 DM", "500-1000 DM", ">= 1000 DM", "unknown/none"]
        )
        credit_history = st.selectbox(
            "Credit history",
            ["no credits taken", "all paid duly (this bank)", "existing credits paid duly",
             "delay in past", "critical/other credits existing"]
        )
        employment_since = st.selectbox(
            "Present employment since",
            ["unemployed", "< 1 year", "1-4 years", "4-7 years", ">= 7 years"]
        )
        housing = st.selectbox("Housing", ["rent", "own", "for free"])
        job = st.selectbox(
            "Job type",
            ["unemployed/unskilled non-resident", "unskilled resident",
             "skilled employee", "management/self-employed/highly qualified"]
        )
        purpose = st.selectbox(
            "Loan purpose",
            ["car (new)", "car (used)", "furniture/equipment", "radio/TV",
             "domestic appliances", "repairs", "education", "vacation",
             "retraining", "business", "others"]
        )

    with st.expander("Additional fields (defaults are fine for most applicants)"):
        personal_status_sex = st.selectbox(
            "Personal status",
            ["male: single", "male: married/widowed", "male: divorced/separated",
             "female: single", "female: divorced/separated/married"]
        )
        other_debtors = st.selectbox("Other debtors/guarantors", ["none", "co-applicant", "guarantor"])
        property_ = st.selectbox("Property", ["real estate", "building society savings/life insurance",
                                                "car or other", "unknown/none"])
        other_installment_plans = st.selectbox("Other installment plans", ["none", "bank", "stores"])
        telephone = st.selectbox("Registered telephone", ["yes", "none"])
        foreign_worker = st.selectbox("Foreign worker", ["yes", "no"])

    submitted = st.form_submit_button("Predict risk")

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if submitted:
    raw = {
        "checking_account_status": checking_account_status,
        "duration_months": duration_months,
        "credit_history": credit_history,
        "purpose": purpose,
        "credit_amount": credit_amount,
        "savings_account": savings_account,
        "employment_since": employment_since,
        "installment_rate_pct": installment_rate_pct,
        "personal_status_sex": personal_status_sex,
        "other_debtors": other_debtors,
        "present_residence_since": present_residence_since,
        "property": property_,
        "age": age,
        "other_installment_plans": other_installment_plans,
        "housing": housing,
        "existing_credits_count": existing_credits_count,
        "job": job,
        "num_dependents": num_dependents,
        "telephone": telephone,
        "foreign_worker": foreign_worker,
    }
    input_df = pd.DataFrame([raw])
    input_df["credit_amount_per_month"] = input_df["credit_amount"] / input_df["duration_months"]
    input_df["age_bucket"] = pd.cut(input_df["age"], bins=[18, 25, 35, 45, 60, 100],
                                     labels=["18-25", "26-35", "36-45", "46-60", "60+"])

    categorical_cols = [
        "checking_account_status", "credit_history", "purpose", "savings_account",
        "employment_since", "personal_status_sex", "other_debtors", "property",
        "other_installment_plans", "housing", "job", "telephone", "foreign_worker",
        "age_bucket"
    ]
    encoded = pd.get_dummies(input_df, columns=categorical_cols)
    encoded.columns = [clean_col(c) for c in encoded.columns]

    # Align to the exact training column layout — missing dummy columns become 0
    final_row = pd.DataFrame(0, index=[0], columns=model_columns)
    for col in encoded.columns:
        if col in final_row.columns:
            final_row[col] = encoded[col].values

    prob = model.predict_proba(final_row)[0, 1]
    prediction = "Bad Risk (likely default)" if prob >= best_threshold else "Good Risk"

    st.subheader("Result")
    c1, c2 = st.columns(2)
    c1.metric("Predicted default probability", f"{prob:.1%}")
    c2.metric("Decision (tuned threshold)", prediction)

    if prob >= best_threshold:
        st.warning(f"Flagged as **higher risk** — probability of default ({prob:.1%}) is above the model's tuned threshold ({best_threshold:.1%}).")
    else:
        st.success(f"Flagged as **good risk** — probability of default ({prob:.1%}) is below the model's tuned threshold ({best_threshold:.1%}).")

    st.subheader("Why the model made this decision")
    shap_values = explainer.shap_values(final_row)

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.force_plot(explainer.expected_value, shap_values[0], final_row.iloc[0],
                     matplotlib=True, show=False)
    st.pyplot(fig, bbox_inches="tight")
    plt.close(fig)

    # Top contributing features, in plain language
    shap_series = pd.Series(shap_values[0], index=final_row.columns).sort_values(key=abs, ascending=False)
    st.markdown("**Top factors driving this prediction:**")
    for feat, val in shap_series.head(5).items():
        direction = "increased" if val > 0 else "decreased"
        st.markdown(f"- `{feat}` {direction} the predicted risk (impact: {val:+.3f})")

st.markdown("---")
st.caption(
    "Built with XGBoost + SHAP on the UCI German Credit dataset. "
    "This is a portfolio/demo project — not financial advice, and not a real credit decision."
)
