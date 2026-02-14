import os
from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and scaler using absolute paths
model = pickle.load(open(os.path.join(BASE_DIR, 'Rainfall.pkl'), 'rb'))
scale = pickle.load(open(os.path.join(BASE_DIR, 'scale.pkl'), 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        feature_names = [
            'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',
            'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
            'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'RainToday',
            'WindGustDir', 'WindDir9am', 'WindDir3pm', 'year', 'month', 'day'
        ]

        # Safer way: Pull specifically by name from the form
        input_data = []
        for name in feature_names:
            val = request.form.get(name)
            input_data.append(float(val))

        df_input = pd.DataFrame([input_data], columns=feature_names)
        
        # Scale and Predict
        scaled_input = scale.transform(df_input)
        prediction = model.predict(scaled_input)

        if prediction[0] == 1:
            return render_template('chance.html')
        return render_template('nochance.html')

    except Exception as e:
        return f"Error during prediction: {str(e)}", 400

# No app.run() here for Vercel
