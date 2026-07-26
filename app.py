import streamlit as st
import joblib
import pandas as pd
import time

st.set_page_config(page_title="Used Car Price Predictor", layout="centered")

# ---------------------------------------------------------
# Custom CSS — Trust-focused automotive theme
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background-color: #F8FAFC;
}

h1 {
    font-weight: 600;
    color: #1E293B;
}

.stButton>button {
    background-color: #2563EB;
    color: white;
    border-radius: 6px;
    border: none;
    padding: 0.6em 1.5em;
    font-weight: 500;
    font-size: 16px;
}

.stButton>button:hover {
    background-color: #1D4ED8;
    color: white;
}

.trust-banner {
    background-color: #EFF6FF;
    border: 1px solid #DBEAFE;
    border-radius: 6px;
    padding: 10px 16px;
    color: #1E40AF;
    font-size: 14px;
    margin-bottom: 20px;
}

.result-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #2563EB;
    padding: 24px;
    border-radius: 8px;
    margin-top: 20px;
}

.result-label {
    color: #64748B;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.5px;
    margin: 0;
}

.result-price {
    color: #1E293B;
    font-size: 34px;
    font-weight: 700;
    margin: 6px 0;
}

.result-range {
    color: #64748B;
    font-size: 15px;
    margin: 0 0 12px 0;
}

.result-context {
    background-color: #F1F5F9;
    padding: 10px 14px;
    border-radius: 6px;
    color: #334155;
    font-size: 14px;
    margin-top: 12px;
}

.disclaimer {
    color: #94A3B8;
    font-size: 12px;
    margin-top: 16px;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load('car_price_model.pkl')

model = load_model()

# Model performance stats (from training) — used for range + context
MODEL_MAE = 135442
DATASET_AVG_PRICE = 427280 * 2.88  # average selling_price converted to PKR
DATASET_SIZE = 2095
MODEL_R2 = 0.90

# ---------------------------------------------------------
# Header + trust signal
# ---------------------------------------------------------
st.title("Used Car Price Predictor")
st.markdown(
    "<p style='color:#64748B; font-size:16px; margin-top:-8px;'>"
    "Enter your car's details to get an estimated market price.</p>",
    unsafe_allow_html=True
)

st.markdown(
    f"<div class='trust-banner'>📊 Trained on {DATASET_SIZE:,} real car listings "
    f"&nbsp;•&nbsp; R² accuracy: {MODEL_R2:.2f}</div>",
    unsafe_allow_html=True
)

st.divider()

# ---------------------------------------------------------
# Form — grouped logically
# ---------------------------------------------------------
st.subheader("Basic Information")
col1, col2 = st.columns(2)
with col1:
    name = st.selectbox("Brand", ['Maruti', 'Hyundai', 'Ford', 'Tata', 'Volkswagen',
                                    'Renault', 'Toyota', 'Mahindra', 'Chevrolet', 'Honda'])
    year = st.number_input("Manufacturing Year", min_value=1995, max_value=2026, value=2018)
with col2:
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000, step=1000)
    owner = st.selectbox("Ownership", ['First Owner', 'Second Owner', 'Third Owner',
                                         'Fourth & Above Owner', 'Test Drive Car'])

st.subheader("Specifications")
col3, col4 = st.columns(2)
with col3:
    fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG'])
    transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
    seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
with col4:
    engine_cc = st.number_input("Engine (CC)", min_value=600, max_value=5000, value=1200, step=100)
    max_power = st.number_input("Max Power (bhp)", min_value=20.0, max_value=500.0, value=90.0)
    seats = st.number_input("Number of Seats", min_value=2, max_value=10, value=5)

st.subheader("Mileage")
col5, col6 = st.columns(2)
with col5:
    mileage = st.number_input("Mileage", min_value=5.0, max_value=40.0, value=18.0)
with col6:
    mileage_unit = st.selectbox("Mileage Unit", ['kmpl', 'km/kg'])

# ---------------------------------------------------------
# Input sanity warning (out-of-typical-range check)
# ---------------------------------------------------------
warnings = []
if year < 2000 or year > 2026:
    warnings.append("Manufacturing year is outside the typical training range.")
if km_driven > 300000:
    warnings.append("Kilometers driven is unusually high compared to training data.")

st.divider()

# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------
if st.button("Predict Price", use_container_width=True):

    if warnings:
        for w in warnings:
            st.warning(f"⚠️ {w} Prediction may be less reliable.")

    with st.spinner("Calculating estimate..."):
        time.sleep(0.6)  # brief pause for perceived processing

        input_df = pd.DataFrame([{
            'name': name, 'year': year, 'km_driven': km_driven, 'fuel': fuel,
            'seller_type': seller_type, 'transmission': transmission, 'owner': owner,
            'seats': seats, 'max_power (in bph)': max_power, 'Mileage Unit': mileage_unit,
            'Mileage': mileage, 'Engine (CC)': engine_cc
        }])

        prediction = model.predict(input_df)[0]
        low = max(0, prediction - MODEL_MAE)
        high = prediction + MODEL_MAE

        vs_avg = ((prediction - DATASET_AVG_PRICE) / DATASET_AVG_PRICE) * 100
        vs_avg_text = f"{abs(vs_avg):.0f}% {'above' if vs_avg >= 0 else 'below'} the average price in our dataset (Rs. {DATASET_AVG_PRICE:,.0f})"

    st.markdown(f"""
    <div class="result-card">
        <p class="result-label">ESTIMATED PRICE</p>
        <p class="result-price">Rs. {prediction:,.0f}</p>
        <p class="result-range">Typical range: Rs. {low:,.0f} – Rs. {high:,.0f}</p>
        <div class="result-context">📍 This is {vs_avg_text}</div>
        <p class="disclaimer">
            Prices converted from INR to PKR at an approximate rate of 1 INR ≈ 2.88 PKR 
            (July 2026) — exchange rates fluctuate, so treat this as an estimate, not a 
            live conversion. The price range reflects the model's typical error margin 
            (±Rs. {MODEL_MAE:,.0f}), based on evaluation against 419 held-out listings 
            the model had not seen during training.
        </p>
    </div>
    """, unsafe_allow_html=True)
