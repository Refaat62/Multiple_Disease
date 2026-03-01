from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI()

#  CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


diabetes_model = joblib.load("diabetes_final.pkl")
heart_model = joblib.load("heart__final.pkl")
kidney_model = joblib.load("kindey__final.pkl")


# DIABETES

class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    try:
        NewBMI_Underweight = 1 if data.BMI <= 18.5 else 0
        NewBMI_Overweight = 1 if 24.9 < data.BMI <= 29.9 else 0
        NewBMI_Obesity_1 = 1 if 29.9 < data.BMI <= 34.9 else 0
        NewBMI_Obesity_2 = 1 if 34.9 < data.BMI <= 39.9 else 0
        NewBMI_Obesity_3 = 1 if data.BMI > 39.9 else 0
        NewInsulinScore_Normal = 1 if 16 <= data.Insulin <= 166 else 0
        NewGlucose_Low = 1 if data.Glucose <= 70 else 0
        NewGlucose_Normal = 1 if 70 < data.Glucose <= 99 else 0
        NewGlucose_Overweight = 1 if 99 < data.Glucose <= 126 else 0
        NewGlucose_Secret = 1 if data.Glucose > 126 else 0

        user_input = [
            data.Pregnancies, data.Glucose, data.BloodPressure, data.SkinThickness,
            data.Insulin, data.BMI, data.DiabetesPedigreeFunction, data.Age,
            NewBMI_Underweight, NewBMI_Overweight, NewBMI_Obesity_1,
            NewBMI_Obesity_2, NewBMI_Obesity_3, NewInsulinScore_Normal,
            NewGlucose_Low, NewGlucose_Normal, NewGlucose_Overweight, NewGlucose_Secret
        ]

        prediction = diabetes_model.predict([user_input])[0]
        prob = diabetes_model.predict_proba([user_input])[0][1]

        if prob < 0.3:
            level = "خطر منخفض"
        elif prob < 0.6:
            level = "خطر متوسط"
        else:
            level = "خطر مرتفع"

        recommendations = (
            "• مراجعة طبيب الغدد الصماء فورًا\n"
            "• إجراء فحص HbA1c وصيام الدم\n"
            "• مراقبة مستوى السكر يوميًا\n"
            "• اتباع نظام غذائي منخفض السكر والكربوهيدرات\n"
            "• ممارسة الرياضة 30 دقيقة يوميًا\n"
            "• مراقبة ضغط الدم والدهون"
        ) if prediction else (
            "• الحفاظ على نظام غذائي متوازن\n"
            "• ممارسة الرياضة بانتظام\n"
            "• فحص مستوى السكر سنويًا\n"
            "• الحفاظ على وزن صحي"
        )

        report = f"""
تقرير مرض السكري
----------------
احتمالية الإصابة: {prob*100:.2f}%
مستوى الخطر: {level}
النتيجة: {"مصاب بالسكري" if prediction else "غير مصاب بالسكري"}

التوصيات:
{recommendations}
"""
        return {"prediction": int(prediction), "report": report}

    except Exception as e:
        return {"error": str(e)}


# HEART

class HeartInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.post("/predict/heart")
def predict_heart(data: HeartInput):
    try:
        user_input = [
            data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs,
            data.restecg, data.thalach, data.exang, data.oldpeak, data.slope,
            data.ca, data.thal
        ]

        prediction = heart_model.predict([user_input])[0]
        prob = heart_model.predict_proba([user_input])[0][1]

        if prob < 0.3:
            level = "خطر منخفض"
        elif prob < 0.6:
            level = "خطر متوسط"
        else:
            level = "خطر مرتفع"

        recommendations = (
            "• مراجعة طبيب قلب\n"
            "• اتباع نظام غذائي صحي للقلب (قليل الملح والدهون)\n"
            "• ممارسة الرياضة 30 دقيقة يوميًا\n"
            "• مراقبة ضغط الدم والكوليسترول\n"
            "• تجنب التدخين والكحول"
        ) if prediction else (
            "• الحفاظ على نظام غذائي صحي ووزن مناسب\n"
            "• ممارسة الرياضة بانتظام\n"
            "• الفحص الدوري للقلب سنويًا\n"
            "• تجنب التدخين والإفراط في الكحول"
        )

        report = f"""
تقرير مرض القلب
----------------
احتمالية الإصابة: {prob*100:.2f}%
مستوى الخطر: {level}
النتيجة: {"إيجابي لمرض القلب" if prediction else "سلبي لمرض القلب"}

التوصيات:
{recommendations}
"""
        return {"prediction": int(prediction), "report": report}

    except Exception as e:
        return {"error": str(e)}


#KIDNEY

class KidneyInput(BaseModel):
    age: float
    blood_pressure: float
    specific_gravity: float
    albumin: float
    sugar: float
    red_blood_cells: float
    pus_cell: float
    pus_cell_clumps: float
    bacteria: float
    blood_glucose_random: float
    blood_urea: float
    serum_creatinine: float
    sodium: float
    potassium: float
    haemoglobin: float
    packed_cell_volume: float
    white_blood_cell_count: float
    red_blood_cell_count: float
    hypertension: float
    diabetes_mellitus: float
    coronary_artery_disease: float
    appetite: float
    peda_edema: float
    aanemia: float

@app.post("/predict/kidney")
def predict_kidney(data: KidneyInput):
    try:
        user_input = [
            data.age, data.blood_pressure, data.specific_gravity, data.albumin, data.sugar,
            data.red_blood_cells, data.pus_cell, data.pus_cell_clumps, data.bacteria,
            data.blood_glucose_random, data.blood_urea, data.serum_creatinine, data.sodium,
            data.potassium, data.haemoglobin, data.packed_cell_volume, data.white_blood_cell_count,
            data.red_blood_cell_count, data.hypertension, data.diabetes_mellitus,
            data.coronary_artery_disease, data.appetite, data.peda_edema, data.aanemia
        ]

        prediction = kidney_model.predict([user_input])[0]
        prob = kidney_model.predict_proba([user_input])[0][1]

        if prob < 0.3:
            level = "خطر منخفض"
        elif prob < 0.6:
            level = "خطر متوسط"
        else:
            level = "خطر مرتفع"

        recommendations = (
            "• مراجعة طبيب كلى\n"
            "• الحفاظ على الترطيب الجيد\n"
            "• تجنب الملح والبروتين الزائد\n"
            "• مراقبة ضغط الدم ومستوى السكر\n"
            "• إجراء فحوصات الكلى بانتظام"
        ) if prediction else (
            "• الحفاظ على نمط حياة صحي\n"
            "• الفحص الدوري\n"
            "• شرب كميات كافية من الماء\n"
            "• تجنب الملح والأطعمة المصنعة"
        )

        report = f"""
تقرير مرض الكلى
----------------
احتمالية الإصابة: {prob*100:.2f}%
مستوى الخطر: {level}
النتيجة: {"إيجابي لمرض الكلى" if prediction else "سلبي لمرض الكلى"}

التوصيات:
{recommendations}
"""
        return {"prediction": int(prediction), "report": report}

    except Exception as e:
        return {"error": str(e)}