from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and preprocessing tools
model = pickle.load(open("logistic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load encoders
le_dependents = pickle.load(open("le_dependents.pkl", "rb"))
le_education = pickle.load(open("le_education.pkl", "rb"))
le_self_employed = pickle.load(open("le_self_employed.pkl", "rb"))
le_property_area = pickle.load(open("le_property_area.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = request.form['Gender']
        married = request.form['Married']
        dependents = request.form['Dependents']
        education = request.form['Education']
        self_employed = request.form['Self_Employed']
        applicant_income = float(request.form['ApplicantIncome'])
        coapplicant_income = float(request.form['CoapplicantIncome'])
        loan_amount = float(request.form['LoanAmount'])
        loan_amount_term = float(request.form['Loan_Amount_Term'])
        credit_history = float(request.form['Credit_History'])
        property_area = request.form['Property_Area']

        # Binary encodings
        gender = 1 if gender.lower() == "male" else 0
        married = 1 if married.lower() == "yes" else 0

        # Encode categorical variables
        dependents = le_dependents.transform([dependents])[0]
        education = le_education.transform([education])[0]
        self_employed = le_self_employed.transform([self_employed])[0]
        property_area = le_property_area.transform([property_area])[0]

        # Create input dataframe
        input_data = pd.DataFrame([[gender, married, dependents, education,
                                    self_employed, applicant_income, coapplicant_income,
                                    loan_amount, loan_amount_term, credit_history,
                                    property_area]],
                                  columns=['Gender', 'Married', 'Dependents', 'Education',
                                           'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
                                           'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'])

        # Scale and predict
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)