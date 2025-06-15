import streamlit as st
import pandas as pd
import joblib

# ========== Load Model, Scaler, dan Dataset ==========
model = joblib.load('model/random_forest_model.pkl')
scaler = joblib.load('model/standar.pkl')
df = pd.read_csv('dataset/data.csv', sep=';', encoding='utf-8')

# Target kolom
target_col = 'Status'

# Fitur sesuai training
feature_columns = [
    'Marital_status', 'Application_mode', 'Application_order', 'Course',
    'Daytime_evening_attendance', 'Previous_qualification',
    'Previous_qualification_grade', 'Nacionality', 'Mothers_qualification',
    'Fathers_qualification', 'Mothers_occupation', 'Fathers_occupation',
    'Admission_grade', 'Displaced', 'Educational_special_needs', 'Debtor',
    'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder',
    'Age_at_enrollment', 'International', 'Curricular_units_1st_sem_credited',
    'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
    'Curricular_units_1st_sem_without_evaluations', 'Curricular_units_2nd_sem_credited',
    'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade',
    'Curricular_units_2nd_sem_without_evaluations', 'Unemployment_rate',
    'Inflation_rate', 'GDP'
]

categorical_features = [
    'Marital_status', 'Application_mode', 'Course', 'Daytime_evening_attendance',
    'Previous_qualification', 'Nacionality', 'Mothers_qualification',
    'Fathers_qualification', 'Mothers_occupation', 'Fathers_occupation',
    'Displaced', 'Educational_special_needs', 'Debtor',
    'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder', 'International'
]

numerical_features_to_scale = [
    'Application_order', 'Previous_qualification_grade', 'Admission_grade',
    'Age_at_enrollment', 'Curricular_units_1st_sem_credited',
    'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved'
]

# Kolom numerik lain
other_numerical = list(set(feature_columns) - set(categorical_features) - set(numerical_features_to_scale))

# Pastikan kolom kategori bertipe kategori
for col in categorical_features:
    df[col] = df[col].astype('category')

# Tipe integer vs float
int_columns = df.select_dtypes(include='int64').columns.tolist()
float_columns = df.select_dtypes(include='float64').columns.tolist()

# ========== Streamlit UI ==========
st.title("🎓 Student Status Prediction App")
st.write("Isi form berikut untuk memprediksi apakah mahasiswa akan **Dropout, Enrolled, atau Graduate**.")

def user_input_features():
    data = {}

    st.subheader("📝 Masukkan Data Mahasiswa")

    # Input numerik dengan scaling
    for col in numerical_features_to_scale:
        if col in int_columns:
            val = st.number_input(col, value=int(df[col].median()), min_value=int(df[col].min()), max_value=int(df[col].max()), step=1, format="%d")
            data[col] = int(val)
        else:
            val = st.number_input(col, value=float(df[col].median()), min_value=float(df[col].min()), max_value=float(df[col].max()), step=0.1)
            data[col] = float(val)

    # Input numerik biasa (tanpa scaling)
    for col in other_numerical:
        if col in int_columns:
            val = st.number_input(col, value=int(df[col].median()), min_value=int(df[col].min()), max_value=int(df[col].max()), step=1, format="%d")
            data[col] = int(val)
        else:
            val = st.number_input(col, value=float(df[col].median()), min_value=float(df[col].min()), max_value=float(df[col].max()), step=0.1)
            data[col] = float(val)

    # Input kategori
    for col in categorical_features:
        options = list(df[col].cat.categories)
        selected = st.selectbox(col, options)
        data[col] = selected

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ========== Tombol Predict ==========
if st.button("🔍 Predict"):
    # Encode kategori
    for col in categorical_features:
        input_df[col] = pd.Categorical(input_df[col], categories=df[col].cat.categories).codes

    # Scaling fitur tertentu
    input_df[numerical_features_to_scale] = scaler.transform(input_df[numerical_features_to_scale])

    # Susun kolom
    input_df = input_df[feature_columns]

    # Prediksi
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    status_labels = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
    result = prediction[0]

    st.subheader("🎯 Hasil Prediksi:")
    st.success(f"Status mahasiswa: **{result}**")

    st.subheader("📊 Probabilitas:")
    proba_df = pd.DataFrame(prediction_proba, columns=[status_labels[i] for i in range(len(status_labels))])
    st.write(proba_df)
