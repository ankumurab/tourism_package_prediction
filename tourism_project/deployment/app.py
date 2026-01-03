import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
import os

# Set Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")

# Function to load the model
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="nishithworld/tourism_package_prediction",
        filename="best_tourism_package_prediction_model_v1.joblib",
        token=HF_TOKEN
    )
    model = joblib.load(model_path)
    return model

# Load the model
model = load_model()

st.set_page_config(
    page_title="Tourism Package Predictor",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="auto",
)

# Setting up the Web Canvas
st.title("✈️ Wellness Tourism Package Purchase Predictor")
st.markdown("### Enter customer details to predict the likelihood of purchasing the Wellness Tourism Package.")

st.header("Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 90, 30)
    monthly_income = st.number_input("Monthly Income", 0, 1000000, 50000, step=1000)
    duration_of_pitch = st.slider("Duration of Pitch (minutes)", 1, 60, 10)
    number_of_followups = st.slider("Number of Follow-ups", 0, 10, 3)
    number_of_trips = st.slider("Number of Trips Annually", 0, 20, 2)
    number_of_person_visiting = st.slider("Number of People Visiting", 1, 10, 1)
    number_of_children_visiting = st.slider("Number of Children Visiting", 0, 5, 0)
    passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    preferred_property_star = st.selectbox("Preferred Property Star Rating", [1, 2, 3, 4, 5])
    pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business", "Unemployed", "Student"])
    designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP", "Director", "Employee"])
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    own_car = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")


# Create a DataFrame from inputs
input_data = pd.DataFrame({
    'Age': [age],
    'MonthlyIncome': [monthly_income],
    'DurationOfPitch': [duration_of_pitch],
    'NumberOfFollowups': [number_of_followups],
    'NumberOfTrips': [number_of_trips],
    'NumberOfPersonVisiting': [number_of_person_visiting],
    'NumberOfChildrenVisiting': [number_of_children_visiting],
    'CityTier': [city_tier],
    'PreferredPropertyStar': [preferred_property_star],
    'PitchSatisfactionScore': [pitch_satisfaction_score],
    'Gender': [gender],
    'MaritalStatus': [marital_status],
    'Occupation': [occupation],
    'Designation': [designation],
    'ProductPitched': [product_pitched],
    'TypeofContact': [typeof_contact],
    'Passport': [passport],
    'OwnCar': [own_car]
})

if st.button("Predict Purchase"):
    st.markdown("---")
    try:
        prediction_proba = model.predict_proba(input_data)[:, 1]
        threshold = 0.45 # Same as used in training
        prediction = (prediction_proba >= threshold).astype(int)

        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.success(f"#### The customer is LIKELY to purchase the Wellness Tourism Package! (Probability: {prediction_proba[0]:.2f})")
            st.balloons()
        else:
            st.info(f"#### The customer is UNLIKELY to purchase the Wellness Tourism Package. (Probability: {prediction_proba[0]:.2f})")

        st.markdown(f"*Probability of purchase: **{prediction_proba[0]:.2f}***")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please check the input values and try again.")

st.markdown("---")
st.markdown("Created with ❤️ for MLOps Project")
