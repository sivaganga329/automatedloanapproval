import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(
    page_title="AI Loan Approval Dashboard",
    layout="wide"
)

# Load model
model = joblib.load("loan_model.pkl")

# Load dataset
df = pd.read_csv("Dataset.csv")

st.title("🏦 AI Powered Loan Approval System")

st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "Go to",
    ["Loan Prediction","Dataset Dashboard","Bulk Prediction"]
)

# -----------------------------------------
# PAGE 1 : LOAN PREDICTION
# -----------------------------------------

if menu == "Loan Prediction":

    st.header("💰 Loan Eligibility Predictor")

    with st.form("prediction_form"):

        col1,col2,col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender",[0,1])
            married = st.selectbox("Married",[0,1])
            dependents = st.slider("Dependents",0,3)

        with col2:
            education = st.selectbox("Education",[0,1])
            self_emp = st.selectbox("Self Employed",[0,1])
            property_area = st.selectbox("Property Area",[0,1,2])

        with col3:
            credit_history = st.selectbox("Credit History",[0,1])
            loan_term = st.number_input("Loan Term",0.0)
            loan_amount = st.number_input("Loan Amount",0.0)

        applicant_income = st.number_input("Applicant Income",0.0)
        coapp_income = st.number_input("Coapplicant Income",0.0)

        predict = st.form_submit_button("Predict Loan")

    if predict:

        total_income = applicant_income + coapp_income

        data = {
            'Gender':gender,
            'Married':married,
            'Dependents':dependents,
            'Education':education,
            'Self_Employed':self_emp,
            'ApplicantIncome':applicant_income,
            'CoapplicantIncome':coapp_income,
            'LoanAmount':loan_amount,
            'Loan_Amount_Term':loan_term,
            'Credit_History':credit_history,
            'Property_Area':property_area,
            'Total_Income':total_income,
            'Total_Income_log':np.log(total_income+1),
            'LoanAmount_log':np.log(loan_amount+1)
        }

        input_df = pd.DataFrame([data])

        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")
        st.progress(int(probability*100))

        if prediction[0] == 1:
            st.success(f"Loan Approved (Confidence {probability*100:.2f}%)")
        else:
            st.error(f"Loan Not Approved (Confidence {probability*100:.2f}%)")

# -----------------------------------------
# PAGE 2 : DATASET DASHBOARD
# -----------------------------------------

elif menu == "Dataset Dashboard":

    st.header("📊 Dataset Analysis")

    col1,col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(
            df,
            x="ApplicantIncome",
            title="Applicant Income Distribution"
        )
        st.plotly_chart(fig1,use_container_width=True)

    with col2:
        fig2 = px.histogram(
            df,
            x="LoanAmount",
            title="Loan Amount Distribution"
        )
        st.plotly_chart(fig2,use_container_width=True)

    st.subheader("Property Area Distribution")

    fig3 = px.pie(
        df,
        names="Property_Area"
    )
    st.plotly_chart(fig3)

# -----------------------------------------
# PAGE 3 : BULK CSV PREDICTION
# -----------------------------------------

elif menu == "Bulk Prediction":

    st.header("📁 Bulk Loan Prediction")

    file = st.file_uploader("Upload CSV File")

    if file is not None:

        data = pd.read_csv(file)
        st.write("Uploaded Data", data.head())

        # Remove unnecessary columns
        if "Loan_ID" in data.columns:
            data = data.drop(columns=["Loan_ID"])

        if "Loan_Status" in data.columns:
            data = data.drop(columns=["Loan_Status"])

        # Handle missing values
        data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)
        data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mean(), inplace=True)
        data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)

        for col in ['Gender','Married','Dependents','Self_Employed']:
            data[col].fillna(data[col].mode()[0], inplace=True)

        # Fix dependents
        data['Dependents'].replace({'3+':3}, inplace=True)
        data['Dependents'] = data['Dependents'].astype(int)

        # 🔥 FIX: Convert categorical text to numeric
        data['Gender'] = data['Gender'].map({'Male':1, 'Female':0})
        data['Married'] = data['Married'].map({'Yes':1, 'No':0})
        data['Self_Employed'] = data['Self_Employed'].map({'Yes':1, 'No':0})
        data['Education'] = data['Education'].map({'Graduate':1, 'Not Graduate':0})

        data['Property_Area'] = data['Property_Area'].map({
            'Rural':0,
            'Semiurban':1,
            'Urban':2
        })

        # Safety fill
        data.fillna(0, inplace=True)

        # Feature Engineering
        data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
        data['Total_Income_log'] = np.log(data['Total_Income'] + 1)
        data['LoanAmount_log'] = np.log(data['LoanAmount'] + 1)

        # Prediction
        preds = model.predict(data)

        data["Prediction"] = preds

        st.success("Prediction Completed")
        st.write("Prediction Result", data)

        st.download_button(
            "Download Results",
            data.to_csv(index=False),
            "loan_predictions.csv"
        )