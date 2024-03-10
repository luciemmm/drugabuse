import streamlit as st
import pandas as pd
from joblib import load
import numpy as np
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
    rf = model.named_steps['randomforestregressor']
    preprocessed_sample = model.named_steps['columntransformer'].transform(features_df)
    individual_predictions = np.array([tree.predict(preprocessed_sample) for tree in rf.estimators_])
    risks = []
    string_=""
    for tree in individual_predictions:
        if tree[0] > 0.1:
            risks.append(1)
            string_+="🚩"
        else:
            risks.append(0)
            string_+="🍀"
            
    st.write(string_)
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
saved_country=country_name
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
help_services = {
    "USA": "https://www.samhsa.gov/find-help/national-helpline",
    "Canada": "https://www.canada.ca/en/health-canada/services/substance-use/get-help/get-help-with-drug-abuse.html",
    "UK": "https://www.talktofrank.com/",
    "Australia": "https://www.health.gov.au/health-topics/drugs/about-drugs/help-and-support",
    "India": "https://www.aarogyasri.telangana.gov.in/de-addiction-centres.html",
    "South Africa": "https://www.sanca.org.za/",
    "Germany": "https://www.dhs.de/suchthilfe",
    "France": "https://www.drogues-info-service.fr/",
    "Brazil": "https://www.gov.br/saude/pt-br/assuntos/saude-de-a-a-z/drogas/viva-voz",
    "Japan": "https://www.ncnp.go.jp/nimh/yakubutsu/support_center/index.html",
    "Spain": "https://www.pnsd.mscbs.gob.es/",
    "Italy": "https://www.salute.gov.it/portale/salute/p1_5.jsp?lingua=italiano&id=21",
    "Netherlands": "https://www.jellinek.nl/",
    "Sweden": "https://www.folkhalsomyndigheten.se/publicerat-material/publikationsarkiv/a/alkohol-narkotika-dopning-tobak-och-spel-om-anvandning-problem-och-atgarder/",
    "Norway": "https://helsenorge.no/rus-og-avhengighet",
    "Denmark": "https://www.sundhedsstyrelsen.dk/da/sundhed/alkohol-og-rygning/rusmidler",
    "Finland": "https://thl.fi/en/web/alcohol-tobacco-and-addictions",
    "Belgium": "https://www.health.belgium.be/en/health/taking-care-yourself/alcohol-tobacco-drugs",
    "Poland": "https://www.gov.pl/web/zdrowie/narkomania",
    "Portugal": "https://www.sicad.pt/PT/Paginas/default.aspx",
    "Greece": "https://www.okana.gr/",
    "Ireland": "https://www.hse.ie/eng/services/list/5/addiction/",
    "Switzerland": "https://www.sucht.swiss/",
    "Austria": "https://www.suchthilfe.at/",
    "Czech Republic": "https://www.drogy-info.cz/",
    "Hungary": "https://drogfokuszpont.hu/",
    "Romania": "https://www.ana.gov.ro/",
    "Bulgaria": "https://ncnpt-addictions.bg/",
    "Croatia": "https://www.hzjz.hr/sluzba-mentalno-zdravlje-i-prevencija-ovisnosti/",
    "Slovakia": "https://www.uvzsr.sk/index.php?option=com_content&view=article&id=3386&Itemid=117",
    "Slovenia": "https://www.nijz.si/sl/preprecevanje-odvisnosti",
    "Estonia": "https://www.tai.ee/en",
    "Latvia": "https://www.vm.gov.lv/en/ministry/narcology/",
    "Lithuania": "http://www.narkotikukontrole.lt/en/",
    "Iceland": "https://www.landlaeknir.is/um-embaettid/greinar/grein/item38520/",
    "New Zealand": "https://www.health.govt.nz/your-health/healthy-living/addictions/alcohol-and-drug-abuse",
    "Mexico": "https://www.gob.mx/salud/conadic",
    "Russia": "https://www.rosminzdrav.ru/",
    "China": "https://www.nncc626.com/",
    "Singapore": "https://www.ncada.org.sg/",
    "Philippines": "https://www.doh.gov.ph/faqs/What-are-the-drug-rehabilitation-centers-in-the-Philippines",
    "Turkey": "https://www.yesilay.org.tr/en",
    "Egypt": "http://www.emro.who.int/egy/programmes/mental-health-substance-abuse.html",
    "Nigeria": "https://www.ndlea.gov.ng/",
    "Kenya": "https://nacada.go.ke/",
    "Colombia": "https://www.minsalud.gov.co/salud/publica/PENT/Paginas/Proteccion-contra-las-drogas.aspx",
    "Argentina": "https://www.argentina.gob.ar/salud/mental-y-adicciones",
    "Chile": "http://www.senda.gob.cl/",
    "Peru": "https://www.devida.gob.pe/",
    "Venezuela": "http://www.fona.gob.ve/",
    "Malaysia": "https://www.adk.gov.my/en/public/",
    "Thailand": "https://www.ryt9.com/s/prg/2770531",
    "Vietnam": "https://www.moh.gov.vn/",
    "South Korea": "http://www.narcotics.or.kr/eng/index.do",
    "Pakistan": "https://anf.gov.pk/",
    "Bangladesh": "http://www.dnc.gov.bd/",
    "United Arab Emirates": "https://www.mohap.gov.ae/en/services/Pages/362.aspx",
    "Saudi Arabia": "https://www.ncnc.gov.sa/",
    "Iran": "https://www.dchq.ir/",
    "Israel": "https://www.antidrugs.gov.il/",
    "Ukraine": "https://moz.gov.ua/kontrol-za-narkotykamy",
    "Ghana": "https://www.nacoc.gov.gh/",
    "Uganda": "https://www.health.go.ug/programs/mental-health/",
    "Tanzania": "https://www.moh.go.tz/en/non-communicable-diseases",
    "Morocco": "https://www.sante.gov.ma/Pages/Accueil.aspx",
    "Algeria": "http://www.ands.dz/",
    "Indonesia": "https://www.kemkes.go.id/",
    "Kazakhstan": "https://www.gov.kz/memleket/entities/dsm?lang=en",
    "Hong Kong SAR": "https://www.nd.gov.hk/en/index.html"
}
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
    if result =="Lower Risk":
        st.image('safety.jpg', caption='According to our predictive model, you are in the lower risk group for drug abuse. However, being in a lower risk group does not guarantee immunity; proactive measures and lifestyle choices are important to minimize the risk further.')
    else:
        st.image('danger.jpg')
        caption=f'If you or someone you know is struggling with substance abuse, remember that you are not alone, and help is available. Here is a link to help you get started: {help_services[saved_country]}. Please, take care of yourself and seek the support you deserve.'
        st.markdown(caption)

