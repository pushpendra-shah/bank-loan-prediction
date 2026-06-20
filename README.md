# Bank Loan Eligibility Predictor

A web app I built that predicts whether a bank loan should be approved or rejected based on the applicant's details. You enter your income, loan amount, credit score and education, and the model tells you if you're likely to get the loan.

I trained a Random Forest model for this and connected it to a simple web form using FastAPI. When you submit the form, the details go to the backend, the model makes a prediction, and the result shows up with a green (approved) or red (rejected) message.

## What it does
- Predicts loan approval based on income, loan amount, credit score and education
- Uses a Random Forest classifier trained with scikit-learn
- FastAPI backend with a /predict endpoint
- Simple HTML form as the frontend
- Shows the result instantly with color (green = approved, red = rejected)

## Tech I used
Python, FastAPI, Uvicorn, scikit-learn, Pandas, NumPy, Pydantic

## Files in this project
- AI_Model.py - trains the Random Forest model and saves it as loan_model.pkl
- loan_model.pkl - the trained model
- main.py - the FastAPI backend with the prediction logic
- index.html - the frontend form

## How to run it

Install the libraries:
```bash
pip install fastapi uvicorn scikit-learn pandas numpy
```

(Optional) Train the model again:
```bash
python AI_Model.py
```

Start the backend:
```bash
uvicorn main:app --reload
```

Then open index.html in your browser and fill in the details to check eligibility.

## Note
Right now the model is trained on a small sample dataset, mainly to show how the whole thing works end to end. The accuracy would improve a lot with a bigger, real-world loan dataset.

## Things I want to add later
- Train on a real loan dataset for better accuracy
- Show the prediction confidence (probability), not just approved/rejected
- Deploy it online so there's a live demo link
