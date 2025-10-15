from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd


app = Flask(__name__)
# Load model
try:
    model = joblib.load('depression_model.pkl')
    print("Model loaded successfully")
except FileNotFoundError:
    print("The model file was not found. Please check the file path.")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    if model is None:
        return "The model failed to load. Please check the model file."

    # Bidirectional selection control
    gender = request.form.get('gender') == 'male'
    alcohol = request.form.get('alcohol') == 'yes'
    hypertension = request.form.get('hypertension') == 'yes'
    diabetes = request.form.get('diabetes') == 'yes'
    asthma = request.form.get('asthma') == 'yes'
    sleep = request.form.get('sleep') == 'yes'

    # Multiple-choice control
    pir = request.form.get('pir')
    education = request.form.get('education')
    sedentary = request.form.get('sedentary')

    # Fill-in-the-blank control
    hei = float(request.form.get('hei'))
    bmi = float(request.form.get('bmi'))
    albumin = float(request.form.get('albumin'))
    neutrophils = float(request.form.get('neutrophils'))
   
    # Label mapping relationship
    pir_mapping = {
        'low': 0,
        'middle': 1,
        'high': 2,
    }
    pir_numeric = pir_mapping.get(pir, 0)

    education_mapping = {
        'less-than-9th': 0,
        '9-11th': 1,
        'high-school': 2,
        'some-college':3,
        'college-graduate':4,
    }
    education_numeric = education_mapping.get(education, 0)

    sedentary_mapping = {
        'low': 0,
        'middle': 1,
        'high': 2,
    }
    sedentary_numeric = sedentary_mapping.get(sedentary, 0)

    # DataFrame
    input_data = pd.DataFrame({        
        'gender': [1 if gender else 0],
        'alcohol': [1 if alcohol else 0],
        'hypertension': [1 if hypertension else 0],
        'diabetes': [1 if diabetes else 0],
        'asthma': [1 if asthma else 0],
        'sleep': [1 if sleep else 0],
        'pir': [pir_numeric],
        'education': [education_numeric],
        'sedentary': [sedentary_numeric],
        'hei': [hei],
        'bmi': [bmi],
        'albumin': [albumin],
        'neutrophils': [neutrophils],
    })

    prediction = model.predict(input_data)[0]
    risk_level = 'high' if prediction == 1 else 'low'
    return redirect(url_for('index', riskLevel=risk_level))

if __name__ == '__main__':
    app.run(debug=True)