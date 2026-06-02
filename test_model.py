import pandas as pd
import numpy as np
import joblib
import os

print('cwd', os.getcwd())
print('files', os.listdir('.'))

try:
    model = joblib.load('loan_model.pkl')
    print('model loaded', type(model))
except Exception as e:
    print('model load failed', type(e).__name__, e)

try:
    df = pd.read_csv('Dataset.csv')
    print('dataset shape', df.shape)
    print('head\n', df.head(3).to_string())
except Exception as e:
    print('dataset load failed', type(e).__name__, e)

if 'model' in globals() and 'df' in globals() and not df.empty:
    r = df.iloc[0].copy()
    X = r.to_dict()
    X['Total_Income'] = X.get('ApplicantIncome', 0) + X.get('CoapplicantIncome', 0)
    X['Total_Income_log'] = np.log(X['Total_Income'] + 1)
    X['LoanAmount_log'] = np.log(X.get('LoanAmount', 0) + 1)
    expected = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area','Total_Income','Total_Income_log','LoanAmount_log']
    input_df = pd.DataFrame([{k: X.get(k, 0) for k in expected}])
    print('input', input_df.to_dict('records'))
    try:
        pred = model.predict(input_df)
        prob = model.predict_proba(input_df)[0][1] if hasattr(model, 'predict_proba') else None
        print('pred', pred, 'prob', prob)
    except Exception as e:
        print('prediction error', type(e).__name__, e)
