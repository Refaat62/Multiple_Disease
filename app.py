import streamlit as st
import pickle
import os
import io
from fpdf import FPDF
import matplotlib.pyplot as plt
from datetime import datetime
import tempfile
import joblib
from streamlit_option_menu import option_menu

# ----------------- CONFIG -----------------
st.set_page_config(
    page_title="Multiple Disease Prediction",
    layout="wide"
)


BASE_DIR = os.getcwd()

diabetes_model       = joblib.load(os.path.join(BASE_DIR,   'diabetes_final.pkl'))
heart_disease_model  = joblib.load(os.path.join(BASE_DIR,   'heart__final.pkl'))
kidney_disease_model = joblib.load(os.path.join(BASE_DIR,   'kindey__final.pkl'))

# ----------------- Recommendation functions -----------------
def diabetes_recommendations(data, prediction):
    rec = []
    score = 0
    if prediction == 1:
        rec.append(" High risk of diabetes detected.")
        if data["Glucose"] > 200:
            rec.append("• Glucose extremely high — reduce sugars immediately and consult a specialist.")
            score += 3
        elif data["Glucose"] > 140:
            rec.append("• Glucose above normal — start a low-carb diet and monitor regularly.")
            score += 2
        elif data["Glucose"] > 99:
            rec.append("• Prediabetes range — lifestyle modifications recommended.")
            score += 1

        if data["BMI"] > 30:
            rec.append("• BMI indicates obesity — supervised weight loss program recommended.")
            score += 2
        elif data["BMI"] > 25:
            rec.append("• Overweight — increase daily activity.")
            score += 1

        if data["BloodPressure"] > 140:
            rec.append("• High blood pressure — reduce salt and control BP.")
            score += 1

        rec += [
            "• Exercise ≥30 minutes daily.",
            "• Avoid sugary drinks and fast food.",
            "• Schedule follow-up with GP/endocrinologist.",
        ]
    else:
        rec.append(" No diabetes detected.")
        if data["BMI"] > 25:
            rec.append("• Maintain healthy weight — consider diet/exercise.")
            score += 1
        if data["Glucose"] > 120:
            rec.append("• Glucose slightly high — reduce simple carbs and recheck in 1–3 months.")
            score += 1
        rec.append("• Keep balanced diet and annual glucose check.")
    return rec, score


def heart_recommendations(data, prediction):
    rec = []
    score = 0
    if prediction == 1:
        rec.append(" High risk of heart disease detected.")
        if data["chol"] > 240:
            rec.append("• Very high cholesterol — immediate dietary change and lipid profile follow-up.")
            score += 2
        elif data["chol"] > 200:
            rec.append("• Elevated cholesterol — reduce saturated fats.")
            score += 1

        if data["trestbps"] > 150:
            rec.append("• Critical blood pressure — consult cardiologist, monitor daily.")
            score += 2
        elif data["trestbps"] > 130:
            rec.append("• High-normal BP — lifestyle control suggested.")
            score += 1

        if data["age"] > 50:
            rec.append("• Age is a risk factor — regular cardiac checkups suggested.")
            score += 1

        rec += [
            "• Walk 20-30 minutes daily.",
            "• Avoid smoking and manage stress.",
            "• Consider ECG and cardiology follow-up.",
        ]
    else:
        rec.append(" Low risk of heart disease.")
        if data["chol"] > 200:
            rec.append("• Slightly high cholesterol — dietary changes recommended.")
            score += 1
        rec.append("• Maintain active lifestyle and yearly checkups.")
    return rec, score


def kidney_recommendations(data, prediction):
    rec = []
    score = 0
    if prediction == 1:
        rec.append(" Kidney issue suspected.")
        if data["creatinine"] > 1.3:
            rec.append("• Elevated creatinine — avoid high-protein diets and consult nephrologist.")
            score += 2
        if data["blood_pressure"] > 140:
            rec.append("• High BP damages kidneys — control BP strictly.")
            score += 2
        if data.get("blood_urea", 0) > 40:
            rec.append("• Elevated blood urea — further renal function tests recommended.")
            score += 1
        rec += [
            "• Reduce salt intake.",
            "• Stay well hydrated.",
            "• Avoid unnecessary NSAIDs.",
            "• Schedule renal function follow-ups.",
        ]
    else:
        rec.append(" Kidneys appear to be functioning normally.")
        if data["blood_pressure"] > 135:
            rec.append("• Slightly high BP — control it to protect kidneys.")
            score += 1
        rec.append("• Stay hydrated and avoid excessive painkillers.")
    return rec, score


# ----------------- Utility: chat-style display -----------------
def render_chat(recommendations):
    st.markdown(
        """
        <style>
        .chatbox {background:#f6f8fa;padding:12px;border-radius:10px;}
        .bubble {background:#ffffff;color:#000000;padding:10px;border-radius:10px;margin:6px 0;box-shadow:0 1px 2px rgba(0,0,0,0.1);}
        .warn {background:#fff3cd;color:#856404;border:1px solid #ffeeba;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="chatbox">', unsafe_allow_html=True)
    for r in recommendations:
        if r.startswith("⚠️"):
            st.markdown(f'<div class="bubble warn">{r}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bubble">{r}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------- Utility: create charts -----------------
def create_diabetes_charts(data):
    images = {}
    # Bar: Glucose vs BMI
    fig, ax = plt.subplots()
    ax.bar(["Glucose", "BMI"], [data["Glucose"], data["BMI"]])
    ax.set_title("Glucose vs BMI")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    buf.seek(0)
    images["bar_glucose_bmi"] = buf.getvalue()
    plt.close(fig)

    # Pie: Glucose category
    g = data["Glucose"]
    if g <= 70:
        label = "Low"
    elif g <= 99:
        label = "Normal"
    elif g <= 126:
        label = "Prediabetes"
    else:
        label = "High"
    fig2, ax2 = plt.subplots()
    ax2.pie([1], labels=[f"Glucose: {label}"], startangle=90)
    ax2.set_title("Glucose Category")
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png")
    buf2.seek(0)
    images["pie_glucose_cat"] = buf2.getvalue()
    plt.close(fig2)

    return images


def create_heart_charts(data):
    images = {}
    fig, ax = plt.subplots()
    ax.bar(["Cholesterol", "RestBP"], [data["chol"], data["trestbps"]])
    ax.set_title("Cholesterol & Resting BP")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    buf.seek(0)
    images["bar_chol_bp"] = buf.getvalue()
    plt.close(fig)

    cp = int(data.get("cp", 0))
    labels = [f"Type {cp}"]
    fig2, ax2 = plt.subplots()
    ax2.pie([1], labels=labels, startangle=90)
    ax2.set_title("Chest Pain (reported)")
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png")
    buf2.seek(0)
    images["pie_cp"] = buf2.getvalue()
    plt.close(fig2)

    return images


def create_kidney_charts(data):
    images = {}
    fig, ax = plt.subplots()
    ax.bar(["Creatinine", "Urea"], [data.get("creatinine", 0), data.get("blood_urea", 0)])
    ax.set_title("Creatinine vs Blood Urea")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    buf.seek(0)
    images["bar_crea_urea"] = buf.getvalue()
    plt.close(fig)

    bp = data.get("blood_pressure", 0)
    if bp <= 120:
        label = "Normal"
    elif bp <= 140:
        label = "Elevated"
    else:
        label = "High"
    fig2, ax2 = plt.subplots()
    ax2.pie([1], labels=[f"BP: {label}"], startangle=90)
    ax2.set_title("Blood Pressure Status")
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png")
    buf2.seek(0)
    images["pie_bp_status"] = buf2.getvalue()
    plt.close(fig2)

    return images


# ----------------- PDF generation -----------------
def clean_text(text):
    cleaned = ""
    for ch in text:
        try:
            ch.encode("latin-1")
            cleaned += ch
        except Exception:
            cleaned += " "
    return cleaned


def generate_pdf(patient_name, disease_name, inputs_dict, prediction_text,
                 recommendations, charts_dict, risk_score):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, clean_text(f"{disease_name} - Patient Report"),
             ln=True, align="C")

    pdf.set_font("Arial", size=10)
    pdf.ln(4)
    pdf.cell(0, 6,
             clean_text(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"),
             ln=True)

    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Patient Inputs:", ln=True)
    pdf.set_font("Arial", size=10)

    for k, v in inputs_dict.items():
        pdf.cell(0, 6, clean_text(f"- {k}: {v}"), ln=True)

    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Prediction Result:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, clean_text(prediction_text))

    pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, clean_text(f"Risk Score: {risk_score}"), ln=True)

    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Recommendations:", ln=True)
    pdf.set_font("Arial", size=10)

    for r in recommendations:
        pdf.multi_cell(0, 6, clean_text(f"- {r}"))

    pdf.ln(4)

    for title, img_bytes in charts_dict.items():
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name

            pdf.set_font("Arial", "B", 10)
            pdf.cell(0, 6, clean_text(title), ln=True)
            pdf.image(tmp_path, w=170)

            os.remove(tmp_path)
        except Exception as e:
            print("Chart error:", e)

        pdf.ln(6)

    out = pdf.output(dest="S").encode("latin-1")
    return out


# ----------------- Sidebar Navigation -----------------
with st.sidebar:
    selected = option_menu(
        "Multiple Disease Prediction",
        ["Diabetes Prediction", "Heart Disease Prediction", "Kidney Disease Prediction"],
        menu_icon="activity",
        icons=["activity", "heart", "person"],
        default_index=0,
    )

# =====================================================================
#                           DIABETES PAGE
# =====================================================================
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction Using Machine Learning")

    # session_state keys
    if "diab_predicted" not in st.session_state:
        st.session_state["diab_predicted"] = False
        st.session_state["diab_data"] = None
        st.session_state["diab_recs"] = None
        st.session_state["diab_score"] = None
        st.session_state["diab_result_text"] = ""

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input("Number of Pregnancies", value="0")
    with col2:
        Glucose = st.text_input("Glucose Level", value="0")
    with col3:
        BloodPressure = st.text_input("BloodPressure Value", value="0")
    with col1:
        SkinThickness = st.text_input("SkinThickness Value", value="0")
    with col2:
        Insulin = st.text_input("Insulin Value", value="0")
    with col3:
        BMI = st.text_input("BMI Value", value="0")
    with col1:
        DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction Value", value="0")
    with col2:
        Age = st.text_input("Age", value="0")

    if st.button("Diabetes Test Result"):
        try:
            NewBMI_Overweight = 0
            NewBMI_Underweight = 0
            NewBMI_Obesity_1 = 0
            NewBMI_Obesity_2 = 0
            NewBMI_Obesity_3 = 0
            NewInsulinScore_Normal = 0
            NewGlucose_Low = 0
            NewGlucose_Normal = 0
            NewGlucose_Overweight = 0
            NewGlucose_Secret = 0

            bmi_f = float(BMI)
            glu_f = float(Glucose)
            ins_f = float(Insulin)

            if bmi_f <= 18.5:
                NewBMI_Underweight = 1
            elif 18.5 < bmi_f <= 24.9:
                pass
            elif 24.9 < bmi_f <= 29.9:
                NewBMI_Overweight = 1
            elif 29.9 < bmi_f <= 34.9:
                NewBMI_Obesity_1 = 1
            elif 34.9 < bmi_f <= 39.9:
                NewBMI_Obesity_2 = 1
            elif bmi_f > 39.9:
                NewBMI_Obesity_3 = 1

            if 16 <= ins_f <= 166:
                NewInsulinScore_Normal = 1

            if glu_f <= 70:
                NewGlucose_Low = 1
            elif 70 < glu_f <= 99:
                NewGlucose_Normal = 1
            elif 99 < glu_f <= 126:
                NewGlucose_Overweight = 1
            elif glu_f > 126:
                NewGlucose_Secret = 1

            user_input = [
                Pregnancies,
                Glucose,
                BloodPressure,
                SkinThickness,
                Insulin,
                BMI,
                DiabetesPedigreeFunction,
                Age,
                NewBMI_Underweight,
                NewBMI_Overweight,
                NewBMI_Obesity_1,
                NewBMI_Obesity_2,
                NewBMI_Obesity_3,
                NewInsulinScore_Normal,
                NewGlucose_Low,
                NewGlucose_Normal,
                NewGlucose_Overweight,
                NewGlucose_Secret,
            ]
            user_input = [float(x) for x in user_input]
            prediction = diabetes_model.predict([user_input])

            if prediction[0] == 1:
                diabetes_result = "The person has diabetes"
            else:
                diabetes_result = "The person has no diabetes"

            data = {
                "Glucose": float(Glucose),
                "BMI": float(BMI),
                "BloodPressure": float(BloodPressure),
                "Age": float(Age),
            }
            recs, score = diabetes_recommendations(data, int(prediction[0]))

            st.session_state["diab_predicted"] = True
            st.session_state["diab_data"] = data
            st.session_state["diab_recs"] = recs
            st.session_state["diab_score"] = score
            st.session_state["diab_result_text"] = diabetes_result

        except Exception as e:
            st.error(f"Error: {e}")

    if st.session_state["diab_predicted"]:
        st.success(st.session_state["diab_result_text"])
        st.subheader("Recommendation Chat")
        render_chat(st.session_state["diab_recs"])

        charts = create_diabetes_charts(st.session_state["diab_data"])

        patient_name = st.text_input("Patient Name (for report)", value="Anonymous")
        pdf_bytes = generate_pdf(
            patient_name,
            "Diabetes",
            st.session_state["diab_data"],
            st.session_state["diab_result_text"],
            st.session_state["diab_recs"],
            charts,
            st.session_state["diab_score"],
        )

        st.download_button(
            label="Download Diabetes Report (PDF)",
            data=pdf_bytes,
            file_name=f"diabetes_report_{patient_name}.pdf",
            mime="application/pdf",
        )

# =====================================================================
#                           HEART PAGE
# =====================================================================
elif selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction Using Machine Learning")

    if "heart_predicted" not in st.session_state:
        st.session_state["heart_predicted"] = False
        st.session_state["heart_data"] = None
        st.session_state["heart_recs"] = None
        st.session_state["heart_score"] = None
        st.session_state["heart_result_text"] = ""

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input("Age", value="0")
    with col2:
        sex = st.text_input("Sex", value="1")
    with col3:
        cp = st.text_input("Chest Pain Types (0-3)", value="0")
    with col1:
        trestbps = st.text_input("Resting Blood Pressure", value="0")
    with col2:
        chol = st.text_input("Serum Cholesterol in mg/dl", value="0")
    with col3:
        fbs = st.text_input("Fasting Blood Sugar > 120 mg/dl (0/1)", value="0")
    with col1:
        restecg = st.text_input("Resting Electrocardiographic results", value="0")
    with col2:
        thalach = st.text_input("Maximum Heart Rate achieved", value="0")
    with col3:
        exang = st.text_input("Exercise Induced Angina (0/1)", value="0")
    with col1:
        oldpeak = st.text_input("ST depression induced by exercise", value="0")
    with col2:
        slope = st.text_input("Slope of the peak exercise ST segment", value="0")
    with col3:
        ca = st.text_input("Major vessels colored by flourosopy (0-3)", value="0")
    with col1:
        thal = st.text_input(
            "thal: 0 = normal; 1 = fixed defect; 2 = reversable defect", value="0"
        )

    if st.button("Heart Disease Test Result"):
        try:
            user_input = [
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal,
            ]
            user_input = [float(x) for x in user_input]
            prediction = heart_disease_model.predict([user_input])

            if prediction[0] == 1:
                heart_disease_result = "This person is having heart disease"
            else:
                heart_disease_result = "This person does not have any heart disease"

            data = {
                "age": float(age),
                "chol": float(chol),
                "trestbps": float(trestbps),
                "cp": int(float(cp)),
            }
            recs, score = heart_recommendations(data, int(prediction[0]))

            st.session_state["heart_predicted"] = True
            st.session_state["heart_data"] = data
            st.session_state["heart_recs"] = recs
            st.session_state["heart_score"] = score
            st.session_state["heart_result_text"] = heart_disease_result

        except Exception as e:
            st.error(f"Error: {e}")

    if st.session_state["heart_predicted"]:
        st.success(st.session_state["heart_result_text"])
        st.subheader("Recommendation Chat")
        render_chat(st.session_state["heart_recs"])

        charts = create_heart_charts(st.session_state["heart_data"])

        patient_name = st.text_input("Patient Name (for report)", value="Anonymous_heart")
        pdf_bytes = generate_pdf(
            patient_name,
            "Heart Disease",
            st.session_state["heart_data"],
            st.session_state["heart_result_text"],
            st.session_state["heart_recs"],
            charts,
            st.session_state["heart_score"],
        )

        st.download_button(
            label="Download Heart Report (PDF)",
            data=pdf_bytes,
            file_name=f"heart_report_{patient_name}.pdf",
            mime="application/pdf",
        )

# =====================================================================
#                           KIDNEY PAGE
# =====================================================================
else:  # "Kidney Disease Prediction"
    st.title("Kidney Disease Prediction using ML")

    if "kidney_predicted" not in st.session_state:
        st.session_state["kidney_predicted"] = False
        st.session_state["kidney_data"] = None
        st.session_state["kidney_recs"] = None
        st.session_state["kidney_score"] = None
        st.session_state["kidney_result_text"] = ""

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        age = st.text_input("Age", value="0")
    with col2:
        blood_pressure = st.text_input("Blood Pressure", value="0")
    with col3:
        specific_gravity = st.text_input("Specific Gravity", value="1")
    with col4:
        albumin = st.text_input("Albumin", value="0")
    with col5:
        sugar = st.text_input("Sugar", value="0")
    with col1:
        red_blood_cells = st.text_input("Red Blood Cell (0/1)", value="0")
    with col2:
        pus_cell = st.text_input("Pus Cell (0/1)", value="0")
    with col3:
        pus_cell_clumps = st.text_input("Pus Cell Clumps (0/1)", value="0")
    with col4:
        bacteria = st.text_input("Bacteria (0/1)", value="0")
    with col5:
        blood_glucose_random = st.text_input("Blood Glucose Random", value="0")
    with col1:
        blood_urea = st.text_input("Blood Urea", value="0")
    with col2:
        serum_creatinine = st.text_input("Serum Creatinine", value="0")
    with col3:
        sodium = st.text_input("Sodium", value="0")
    with col4:
        potassium = st.text_input("Potassium", value="0")
    with col5:
        haemoglobin = st.text_input("Haemoglobin", value="0")
    with col1:
        packed_cell_volume = st.text_input("Packed Cell Volume", value="0")
    with col2:
        white_blood_cell_count = st.text_input("White Blood Cell Count", value="0")
    with col3:
        red_blood_cell_count = st.text_input("Red Blood Cell Count", value="0")
    with col4:
        hypertension = st.text_input("Hypertension (0/1)", value="0")
    with col5:
        diabetes_mellitus = st.text_input("Diabetes Mellitus (0/1)", value="0")
    with col1:
        coronary_artery_disease = st.text_input("Coronary Artery Disease (0/1)", value="0")
    with col2:
        appetite = st.text_input("Appetite (0/1)", value="1")
    with col3:
        peda_edema = st.text_input("Peda Edema (0/1)", value="0")
    with col4:
        aanemia = st.text_input("Anemia (0/1)", value="0")

    if st.button("Kidney's Test Result"):
        try:
            user_input = [
                age,
                blood_pressure,
                specific_gravity,
                albumin,
                sugar,
                red_blood_cells,
                pus_cell,
                pus_cell_clumps,
                bacteria,
                blood_glucose_random,
                blood_urea,
                serum_creatinine,
                sodium,
                potassium,
                haemoglobin,
                packed_cell_volume,
                white_blood_cell_count,
                red_blood_cell_count,
                hypertension,
                diabetes_mellitus,
                coronary_artery_disease,
                appetite,
                peda_edema,
                aanemia,
            ]
            user_input = [float(x) for x in user_input]
            prediction = kidney_disease_model.predict([user_input])

            if prediction[0] == 1:
                kidney_diagnosis = "The person has Kidney's disease"
            else:
                kidney_diagnosis = "The person does not have Kidney's disease"

            data = {
                "creatinine": float(serum_creatinine),
                "blood_pressure": float(blood_pressure),
                "blood_urea": float(blood_urea),
            }
            recs, score = kidney_recommendations(data, int(prediction[0]))

            st.session_state["kidney_predicted"] = True
            st.session_state["kidney_data"] = data
            st.session_state["kidney_recs"] = recs
            st.session_state["kidney_score"] = score
            st.session_state["kidney_result_text"] = kidney_diagnosis

        except Exception as e:
            st.error(f"Error: {e}")

    if st.session_state["kidney_predicted"]:
        st.success(st.session_state["kidney_result_text"])
        st.subheader("Recommendation Chat")
        render_chat(st.session_state["kidney_recs"])

        charts = create_kidney_charts(st.session_state["kidney_data"])

        patient_name = st.text_input("Patient Name (for report)", value="Anonymous_kidney")
        pdf_bytes = generate_pdf(
            patient_name,
            "Kidney Disease",
            st.session_state["kidney_data"],
            st.session_state["kidney_result_text"],
            st.session_state["kidney_recs"],
            charts,
            st.session_state["kidney_score"],
        )

        st.download_button(
            label="Download Kidney Report (PDF)",
            data=pdf_bytes,
            file_name=f"kidney_report_{patient_name}.pdf",
            mime="application/pdf",
        )
