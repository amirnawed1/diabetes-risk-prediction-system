import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# ========== HEADER ==========
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("🩺")
with col2:
    st.title("Diabetes Risk Prediction System")
    st.markdown("### Assess diabetes risk using key health indicators")
st.markdown("---")

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    with open("diabetes_model.pkl", "rb") as file:
        model = pickle.load(file)
    # Fix for older scikit-learn compatibility
    if hasattr(model, "named_steps"):
        if "classifier" in model.named_steps:
            model.named_steps["classifier"].multi_class = "ovr"
    return model

model = load_model()

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("🩻 Patient Health Data")
    st.markdown("Adjust the values based on medical reports")

    pregnancies = st.number_input("Pregnancies", 0, 20, 1, help="Number of times pregnant")
    glucose = st.slider("Glucose Level (mg/dL)", 0, 250, 120, help="Plasma glucose concentration")
    blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 150, 70, help="Diastolic blood pressure")
    skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20, help="Triceps skinfold thickness")
    insulin = st.slider("Insulin (mu U/ml)", 0, 900, 80, help="2-Hour serum insulin")
    bmi = st.slider("BMI", 0.0, 70.0, 25.0, help="Body mass index (weight in kg/(height in m)^2)")
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, help="Diabetes mellitus history in relatives")
    age = st.slider("Age (years)", 1, 120, 30)

    st.markdown("---")
    predict_btn = st.button("🔍 Assess Diabetes Risk", type="primary", use_container_width=True)

# ========== MAIN AREA ==========
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Key Health Indicators")
    st.metric("Glucose", f"{glucose} mg/dL")
    st.metric("BMI", f"{bmi:.1f} kg/m²")
    st.metric("Age", f"{age} years")

with col2:
    st.markdown("### ℹ️ Healthy Reference Ranges")
    st.info("""
    - **Glucose:** 70-100 mg/dL (fasting)
    - **Blood Pressure:** < 120/80 mm Hg
    - **BMI:** 18.5-24.9
    """)

# ========== PREDICTION ==========
if predict_btn:
    st.markdown("---")
    st.markdown("## 🎯 Risk Assessment Result")

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, diabetes_pedigree, age]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Risk gauge
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.barh(0, 1, color="#E0E0E0", height=0.5)
    ax.barh(0, probability, color="#FF6B6B" if probability > 0.5 else "#4ECDC4", height=0.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Risk Probability")
    ax.set_yticks([])
    st.pyplot(fig)

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        if prediction == 1:
            st.error(f"### ⚠️ High Risk of Diabetes\nProbability: **{probability:.1%}**")
        else:
            st.success(f"### ✅ Low Risk of Diabetes\nProbability: **{probability:.1%}**")

    with res_col2:
        st.markdown("#### Recommended Actions")
        if prediction == 1:
            st.markdown("""
            - Consult a doctor immediately
            - Monitor blood sugar regularly
            - Improve diet & exercise
            """)
        else:
            st.markdown("""
            - Maintain healthy lifestyle
            - Regular check-ups
            - Keep weight in check
            """)

# ========== FOOTER ==========
st.markdown("---")
st.markdown("### 👨‍💻 About This Project")
st.markdown("""
**Model:** Logistic Regression with scaling pipeline  
**Data:** PIMA Indians Diabetes Database  
**Built by:** Amir Nawed | [GitHub](https://github.com/amirnawed1)  
**Deployed on:** Streamlit Cloud  
**Disclaimer:** Educational project only – not for real medical diagnosis.
""")
