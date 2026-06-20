import pickle
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


with open("loan_model.pkl", "rb") as f:
    loan_model = pickle.load(f)



class LoanInput(BaseModel):
    income: float
    loan_amount: float
    credit_score: float
    education: int  # 1 for Graduate, 0 for Not Graduate


@app.get("/")
def home():
    return {"message": "Loan Prediction API Active!"}


@app.post("/predict")
def predict_loan(data: LoanInput):
    input_data = np.array(
        [[data.income, data.loan_amount, data.credit_score, data.education]]
    )

    # Model se prediction lena
    prediction = loan_model.predict(input_data)

    if prediction[0] == 1:
        status = "Approved 🎉 (Loan Diya Jaa Sakta Hai)"
        color = "#4CAF50"
    else:
        status = "Rejected ❌ (Loan Nahi Diya Jaa Sakta - Risk High Hai)"
        color = "#f44336"

    return {"loan_status": status, "color": color}
