import streamlit as st
import pandas as pd
from joblib import load
#from job import DummyClassifier
countries=["Belgium","Finland","France","Germany","Hungary", "Italy", "Lithuania", "Netherlands", "Norway", "Portugal", "Sweden"]
# Load the trained model (make sure to include the correct path to your .joblib file)
def predict_risk(input_data, model, original_data):
    """
    Make a prediction using the Random Forest Regressor model for the given input data.
    Automatically retrieves country-specific data based on the 'Country' field in the input.

    Parameters:
    - input_data: A dictionary containing the input data for the non-country-specific features.
    - model: The trained Random Forest Regressor model.
    - original_data: The original dataset from which to retrieve country-specific values.

    Returns:
    - The predicted risk.
    """

    # Validate that the necessary inputs are in the input_data
    required_inputs = ['Country', 'Sex', 'Age_Group', 'Income', 'Happiness', 'Mental_Health']
    for required_input in required_inputs:
        if required_input not in input_data:
            raise ValueError(f"Missing required input: {required_input}")

    # Retrieve the country-specific data
    country_data = original_data[original_data['Country'] == input_data['Country']].iloc[0]

    # Construct the full feature set for prediction
    features = input_data.copy()
    country_specific_columns = ['Enrollment', 'Unemployment', 'Decriminalization',
                                'Imprisonment days', 'Drug law offences',
                                'Number of sites with NSP', 'Total Direct Expenditure']
    for col in country_specific_columns:
        features[col] = country_data[col]

    # Convert the features to a DataFrame in the same format as the training data
    features_df = pd.DataFrame([features])
    print("Features data after transformation:", features_df)

    # Make a prediction
    prediction = model.predict(features_df)

    return prediction

# Function to predict and return the result
def predict_outcome(probability):
    if probability >= 0.1:  # Assuming that '1' indicates "Danger"
        return "Higher Risk"
    else:
        return "Lower Risk"

# Streamlit UI components
st.title("Drug Abuse Vulnerability Prediction ")
from streamlit.components.v1 import html

# Define your text and marquee speed
marquee_text = "🐜🐜🐜🐜🐜🐜🐜🐜🐜🐜🐜🐜🐜🐜🐜🐜"
marquee_speed = 10  # Adjust this value to control the speed of the marquee

# HTML and CSS to create the marquee effect
marquee_html = f"""
<div style="overflow-x: hidden; white-space: nowrap; font-size: 30px; width: 100%;">
  <div style="display: inline-block; padding-left:100%; animation: marquee {marquee_speed}s linear infinite;">
    {marquee_text}
  </div>
</div>
<style>
@keyframes marquee {{
  from {{ transform: translate(0, 0); }}
  to {{ transform: translate(-100%, 0); }}
}}
</style>
"""

# Display the marquee in Streamlit
html(marquee_html)
# Input fields
default_country="Germany"
country_name = st.text_input("Country Name")
if country_name not in countries:
    country_name=default_country

# Check if the country_name is not empty and is in the country_happiness dictionary

sex_dict={"Male":"1","Female":"2","Non-binary":"1","Prefer not to disclose":"1"}
sex = st.selectbox("Gender", ["Male", "Female", "Non-binary","Prefer not to disclose"])

#school = st.selectbox("School", ["Primary", "Secondary", "Higher","Not in School","Prefer not to disclose"])

economic_status = st.number_input("Yearly Income in USD", min_value=0, max_value=None, step=1000, format="%d")
mental_health = st.number_input("How tired do you feel mentally overall on a scale from 0 to 100?", min_value=0, max_value=100, step=1, format="%d")
happiness_index = st.number_input("How happy do you feel overall on a scale from 0 to 100?", min_value=0, max_value=100, step=1, format="%d")

age_groups = [
    "0 year",
    "1 year",
    "2 years",
    "3 years",
    "4 years",
    "5-9 years",
    "10-14 years",
    "15-19 years",
    "20-24 years",
    "25-29 years",
    "30-34 years",
    "35-39 years",
    "40-44 years",
    "45-49 years",
    "50-54 years",
    "55-59 years",
    "60-64 years",
    "65-69 years",
    "70-74 years",
    "75-79 years",
    "80-84 years",
    "85-89 years",
    "90-94 years",
    "95 years and above"
]
age_dict = {age_group: str(age_group_index + 2) for age_group_index, age_group in enumerate(age_groups)}
# Use selectbox for user input
age_group = st.selectbox("Select Age Group", age_groups)
# Check for 'Prefer not to disclose' option in mental_health
if mental_health == "Prefer not to disclose":
    mental_health = 5

# Check for 'Prefer not to disclose' option in sex
if sex == "Prefer not to disclose":
    sex = "Male"
elif sex == "Non-binary":
    sex = "Male"

data = pd.read_csv('for_analyses4.csv')
loaded_model = load('model.joblib')

if st.button("Predict"):
    # Prepare user inputs for the model
    # This is a placeholder; you'll need to adjust it according to your model's needs
    input_data = {
    'Country': country_name,
    'Sex': sex_dict[sex],
    'Age_Group': age_dict[age_group],
    'Income': economic_status,
    'Happiness': happiness_index,
    'Mental_Health': mental_health
    }
    print(input_data)

    # Get prediction
    probability = predict_risk(input_data, loaded_model, data)
    result= predict_outcome(probability)
    
    # Display the result
    st.write(f"The prediction is: **{result}**")
    st.write(f"Risk probability: **{probability}**")
