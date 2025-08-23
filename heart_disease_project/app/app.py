import streamlit as st
import pandas as pd
import joblib
import pymongo
from datetime import datetime

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["heart_disease_db"]
collection = db["predictions"]

st.set_page_config(page_title="Heart Disease Risk Assessment")
st.title("Heart Disease Risk Assessment 🫀")
st.write("Demo — not a medical diagnosis.")

model = joblib.load("models/heart_rf_pipeline.joblib")

age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex", ["Male","Female"])
cp = st.selectbox("Chest pain type (cp)", [1,2,3,4])
trestbps = st.number_input("Resting BP", 80, 250, 130)
chol = st.number_input("Cholesterol", 100, 600, 250)
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", [0,1])
restecg = st.selectbox("Resting ECG (0,1,2)", [0,1,2])
thalach = st.number_input("Max heart rate achieved", 50, 250, 150)
exang = st.selectbox("Exercise induced angina", [0,1])
oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox("Slope (1-3)", [1,2,3])
ca = st.selectbox("Number of major vessels (0-3)", [0,1,2,3])
thal = st.selectbox("Thal (3,6,7)", [3,6,7])

sex_num = 1 if sex == "Male" else 0
input_data = {
    'age': age, 'sex': sex_num, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
    'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
    'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
}
input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    prob = model.predict_proba(input_df)[:,1][0]
    pred = model.predict(input_df)[0]

    st.metric("Risk probability", f"{prob:.2f}")
    st.write("Predicted class:", "High risk" if pred==1 else "Low risk")
    if prob >= 0.5:
        st.warning("Predicted higher risk — clinical follow-up recommended.")
    else:
        st.success("Predicted lower risk.")

    record = {
        **input_data,
        "prediction": int(pred),
        "probability": float(prob),
        "timestamp": datetime.utcnow()
    }
    collection.insert_one(record)
    st.info("✅ Prediction saved to database!")
