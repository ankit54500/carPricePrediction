from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import warnings
import sys

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Try to load the model with error handling
try:
    with open("LinearRegressionModel.pkl", 'rb') as f:
        modelLr = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check if the model file exists and is compatible with your sklearn version")
    sys.exit(1)

# Load the data
try:
    car = pd.read_csv('CleanedData.csv')
    print("Data loaded successfully!")
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_type = sorted(car['fuel_type'].unique())
    return render_template('index.html', companies=companies, models=models, years=years, fuel_types=fuel_type)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        company = request.form.get('company')
        model = request.form.get('model')  # Fixed: was 'car_model' but HTML sends 'model'
        year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        kms_driven = int(request.form.get('kms_driven'))
        
        print(f"Received data: {company}, {model}, {year}, {fuel_type}, {kms_driven}")
        
        # Create prediction dataframe
        prediction_data = pd.DataFrame([[model, company, year, kms_driven, fuel_type]], 
                                     columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
        
        # Make prediction
        prediction = modelLr.predict(prediction_data)
        result = str(np.round(prediction[0], 2))
        
        print(f"Prediction result: {result}")
        return result
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Error: Could not make prediction"

if __name__ == "__main__":
    app.run(debug=True)