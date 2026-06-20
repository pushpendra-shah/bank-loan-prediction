import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = {
    "Income": [50000, 20000, 80000, 30000, 90000, 15000, 60000, 25000],
    "LoanAmount": [15000, 25000, 10000, 30000, 50000, 10000, 20000, 5000],
    "CreditScore": [750, 550, 800, 600, 780, 500, 710, 580],
    "Education": [1, 0, 1, 1, 1, 0, 1, 0],
    "Loan_Status": [1, 0, 1, 0, 1, 0, 1, 1],  # 1 = Approved, 0 = Rejected
}

df = pd.DataFrame(data)

X = df[["Income", "LoanAmount", "CreditScore", "Education"]]
y = df["Loan_Status"]

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

with open("loan_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("AI Model kamyabi se train aur save ho gaya hai! (loan_model.pkl ban gayi hai)")
