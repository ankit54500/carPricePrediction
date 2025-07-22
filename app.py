from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import warnings
import sys
import sklearn
import os


# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)


# Try to load the model with error handling and version compatibility
def load_model_safely():
    try:
        # First, try loading normally
        with open("LinearRegressionModel.pkl", 'rb') as f:
            modelLr = pickle.load(f)
        print("Model loaded successfully!")
        return modelLr
    except Exception as e:
        print(f"Error loading model with current sklearn version: {e}")
        print("Attempting compatibility fixes...")
        
        try:
            # Try with different protocol
            with open("LinearRegressionModel.pkl", 'rb') as f:
                modelLr = pickle.load(f)
            return modelLr
        except:
            print("Failed to load model. You need to retrain the model with current sklearn version.")
            return None

# Load the model
modelLr = load_model_safely()

if modelLr is None:
    print("SOLUTION: Please retrain your model with the current sklearn version")
    print("Check the solution code below for model retraining script")

# Load the data
try:
    car = pd.read_csv('CleanedData.csv')
    print("Data loaded successfully!")
    print(f"Dataset shape: {car.shape}")
    print(f"Columns: {list(car.columns)}")
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
    if modelLr is None:
        return "Error: Model not loaded. Please retrain the model."
    
    try:
        # Get form data
        company = request.form.get('company')
        model = request.form.get('model')
        year = request.form.get('year')
        fuel_type = request.form.get('fuel_type')
        kms_driven = request.form.get('kms_driven')
        
        # Validate inputs
        if not all([company, model, year, fuel_type, kms_driven]):
            return "Error: Missing required fields"
        
        # Convert to appropriate types
        try:
            year = int(year)
            kms_driven = int(kms_driven)
        except ValueError:
            return "Error: Invalid year or kilometers value"
        
        
        # Validate data against dataset
        if company not in car['company'].values:
            return f"Error: Company '{company}' not found in dataset"
        
        if model not in car['name'].values:
            return f"Error: Model '{model}' not found in dataset"
        
        if fuel_type not in car['fuel_type'].values:
            return f"Error: Fuel type '{fuel_type}' not found in dataset"
        
        # Create prediction dataframe with same column order as training
        prediction_data = pd.DataFrame([[model, company, year, kms_driven, fuel_type]], 
                                     columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
        
        
        
        # Make prediction
        prediction = modelLr.predict(prediction_data)
        result = str(np.round(prediction[0], 2))
        
        
        return result
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

# Add a test route to check model status
@app.route('/model-status')
def model_status():
    if modelLr is None:
        return "Model: Not loaded"
    else:
        return f"Model: Loaded successfully (sklearn version: {sklearn.__version__})"

if __name__ == "__main__":
    if modelLr is not None:
        app.run(debug=True)
    else:
        print("\n" + "="*50)
        print("MODEL LOADING FAILED!")
        print("="*50)
        print("Please run the model retraining script first.")
        print("Check the 'Model Retraining Script' artifact for the solution.")


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))