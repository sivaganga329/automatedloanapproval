import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Dataset.csv")

df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

for col in ['Gender','Married','Dependents','Self_Employed']:
    df[col].fillna(df[col].mode()[0], inplace=True)

df['Dependents'].replace({'3+':3}, inplace=True)
df['Dependents'] = df['Dependents'].astype(int)

df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['Total_Income_log'] = np.log(df['Total_Income'] + 1)
df['LoanAmount_log'] = np.log(df['LoanAmount'] + 1)

encoder = LabelEncoder()

for col in ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']:
    df[col] = encoder.fit_transform(df[col])

X = df.drop(columns=['Loan_ID','Loan_Status'])
y = df['Loan_Status']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestClassifier(n_estimators=150)

model.fit(X_train,y_train)

joblib.dump(model,"loan_model.pkl")

print("Model trained and saved!")