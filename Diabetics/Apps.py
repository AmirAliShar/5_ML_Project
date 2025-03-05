import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("Diabetics/diabetes.csv")

# Load Model & Scaler
with open("Diabetics/diabetes_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)


# Streamlit Sidebar for Navigation
st.sidebar.title("Exploratory Data Analysis")
option = st.sidebar.radio("Select Option", ["EDA", "Diabetes Prediction"])

# ----------------------------------
# 📌 OPTION 1: Exploratory Data Analysis (EDA)
# ----------------------------------
if option == "EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")
    
    # Show dataset
    st.write("### Dataset Overview")
    st.write(df.head())

    st.write("Dataset Shape")
    st.write(df.shape)

    # Display dataset statistics
    st.write("### Dataset Statistics")
    st.write(df.describe().T)

    # Class distribution
    st.write("### Diabetes Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Outcome", data=df, ax=ax)
    st.pyplot(fig)

    # Pie Chart of Outcome
    st.write("### Pie Chart of Diabetes Outcome")
    outcome_counts = df["Outcome"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(outcome_counts, labels=['No Diabetes', 'Diabetes'], autopct='%1.1f%%', startangle=90, colors=['skyblue', 'green'])
    ax.set_title("Pie Chart of Outcome")
    st.pyplot(fig)

    # Scatter Plot of Glucose vs. BMI with Regression Line
    st.write("### Scatter Plot: Glucose vs. BMI")
    fig = sns.lmplot(x='Glucose', y='BMI', hue='Outcome', data=df, aspect=1.5, markers=["o", "s"])
    plt.title("Glucose vs. BMI")
    st.pyplot(fig)

    # Line Plot of Age vs. Glucose
    st.write("### Line Plot: Age vs. Glucose")
    df_sorted = df.sort_values("Age")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_sorted["Age"], df_sorted["Glucose"], marker='o', color='b')
    ax.set_title("Age vs. Glucose")
    ax.set_xlabel("Age")
    ax.set_ylabel("Glucose")
    st.pyplot(fig)

    # Distribution of Features
    st.write("### Feature Distributions")
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 8))
    axes = axes.flatten()
    for i, column in enumerate(df.columns[:-1]):  # Excluding 'Outcome'
        sns.histplot(df[column], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {column}')

    plt.tight_layout()
    st.pyplot(fig)

     # Correlation Heatmap
    st.write("### Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    
    


# ----------------------------------
# 📌 OPTION 2: Diabetes Prediction
# ----------------------------------
elif option == "Diabetes Prediction":
    st.title("🩺 Diabetes Prediction")
    st.write("Enter patient details to predict diabetes.")

    # User Inputs
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=180, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.2f")
    age = st.number_input("Age", min_value=0, max_value=100, value=30)

    # Prediction Button
    if st.button("Predict Diabetes"):
        # Prepare input data
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        
        # Make prediction
        prediction = model.predict(input_data)
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        
        st.success(f"Prediction: **{result}**")
        

# ----------------------------------

# Run this file using:
# open the window shell and below without #
# streamlit run Apps.py
