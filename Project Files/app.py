from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved model and scaler
model = pickle.load(open('Rainfall.pkl', 'rb'))
scale = pickle.load(open('scale.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from the HTML form
    input_features = [float(x) for x in request.form.values()]
    features_array = [np.array(input_features)]
    
    # Column names must match exactly what we used in training
    feature_names = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',
                     'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
                     'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'RainToday',
                     'WindGustDir', 'WindDir9am', 'WindDir3pm', 'year', 'month', 'day']
    
    # Convert to DataFrame
    df_input = pd.DataFrame(features_array, columns=feature_names)
    
    # Scale the input
    scaled_input = scale.transform(df_input)
    
    # Make prediction
    prediction = model.predict(scaled_input)
    
    if prediction[0] == 1:
        return render_template('chance.html')
    else:
        return render_template('nochance.html')

if __name__ == "__main__":
    app.run(debug=True)