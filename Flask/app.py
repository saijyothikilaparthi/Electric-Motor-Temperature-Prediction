from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load Model and Scaler
try:
    with open('model.save', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: model.save not found.")
    model = None

try:
    with open('transform.save', 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully.")
except FileNotFoundError:
    print("Error: transform.save not found.")
    scaler = None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/manual')
def manual_predict():
    return render_template('Manual_predict.html')

@app.route('/sensor')
def sensor_predict():
    return render_template('Sensor_predict.html')

@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or Scaler not loaded properly.'})
    
    try:
        # Get input values from form
        ambient = float(request.form['ambient'])
        coolant = float(request.form['coolant'])
        u_d = float(request.form['u_d'])
        u_q = float(request.form['u_q'])
        motor_speed = float(request.form['motor_speed'])
        i_d = float(request.form['i_d'])
        i_q = float(request.form['i_q'])
        
        # Create input array
        features = np.array([[ambient, coolant, u_d, u_q, motor_speed, i_d, i_q]])
        
        # Scale the input
        scaled_features = scaler.transform(features)
        
        # Predict
        prediction = model.predict(scaled_features)
        output = round(prediction[0], 2)
        
        return render_template('Manual_predict.html', prediction_text=f'Predicted Motor Temperature (PM): {output} °C')
        
    except Exception as e:
        return render_template('Manual_predict.html', prediction_text=f'Error: {str(e)}')

@app.route('/predict_sensor', methods=['POST'])
def predict_sensor():
    if model is None or scaler is None:
        return render_template('Sensor_predict.html', prediction_text='Error: Model or Scaler not loaded properly.')

    try:
        if 'sensor_data' not in request.files:
            return render_template('Sensor_predict.html', prediction_text='No file part')
        
        file = request.files['sensor_data']
        if file.filename == '':
            return render_template('Sensor_predict.html', prediction_text='No selected file')
            
        if file:
            # Read CSV
            df = pd.read_csv(file)
            
            # Ensure we have the correct columns (ignoring profile_id etc if present)
            # Expected features order: ambient, coolant, u_d, u_q, motor_speed, i_d, i_q
            # We take the columns by name if they exist, or just the first 7 columns if names match expectation
            
            # Simple approach: Taking the FIRST row for demonstration as per the image imply single result
            # Or usually "Batch" means return a CSV. 
            # But the user image shows: "Sensor data given... {{ data }}" and a SINGLE "Prediction".
            # So we will take the first row of the uploaded CSV.
            
            # Extract features from the first row
            first_row = df.iloc[0]
            
            # Assuming columns are correct in CSV. 
            # Features required: ambient, coolant, u_d, u_q, motor_speed, i_d, i_q
            # We select these specifically to avoid 'profile_id' etc.
            
            feature_cols = ['ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'i_d', 'i_q']
            
            # check if columns exist
            if not all(col in df.columns for col in feature_cols):
                 return render_template('Sensor_predict.html', prediction_text=f'Error: CSV must contain columns: {feature_cols}')

            input_data = first_row[feature_cols].values.reshape(1, -1)
            
            # Scale
            scaled_data = scaler.transform(input_data)
            
            # Predict
            prediction = model.predict(scaled_data)
            output = round(prediction[0], 2)
            
            # Format data for display (list of values)
            display_data = input_data[0].tolist()
            # Round for display
            display_data = [round(x, 2) for x in display_data]
            
            return render_template('Sensor_predict.html', 
                                   data=str(display_data), 
                                   prediction_text=f'Predicted Motor Temperature (PM): {output} °C')
            
    except Exception as e:
        return render_template('Sensor_predict.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
