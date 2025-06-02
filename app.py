
import streamlit as st
import joblib
import pandas as pd

# Load trained GBM model
model = joblib.load("final_gb_model.pkl")

st.set_page_config(page_title="RWRS² Calculator", layout="centered")

st.title("🏈 RWRS² Rookie WR Success Score")
st.markdown("Predict the success of a rookie WR using the trained GBM model (scale: 0.0 = bust, 1.0 = elite).")

# User inputs
draft_round = st.slider("Draft Round", 1, 7, 1)
college_conference = st.selectbox("College Conference", ["SEC", "Big Ten", "Big 12", "ACC", "Pac-12", "Other"])
age_entering_nfl = st.slider("Age Entering NFL", 20, 25, 21)

if st.button("Calculate RWRS² Score"):
    input_df = pd.DataFrame({
        "Draft Round": [draft_round],
        "College Conference": [college_conference],
        "Age Entering NFL": [age_entering_nfl]
    })
    prediction = model.predict(input_df)[0]
    st.metric("Predicted RWRS² Score", f"{prediction:.2f}")
