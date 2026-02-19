import pickle
import numpy as np
import pandas as pd

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

if model and scaler:
    # Sample Input from CSV Row 2
    # ambient,coolant,u_d,u_q,motor_speed,i_d,i_q
    # 27.483570765056164,63.99355436586002,-33.758913748719074,-7.23422673632173,2136.506395035066,-133.42244074699263,123.5555524693113
    
    ambient = 27.48
    coolant = 63.99
    u_d = -33.76
    u_q = -7.23
    motor_speed = 2136.51
    i_d = -133.42
    i_q = 123.56
    
    features = np.array([[ambient, coolant, u_d, u_q, motor_speed, i_d, i_q]])
    
    # Scale
    scaled_features = scaler.transform(features)
    
    # Predict
    prediction = model.predict(scaled_features)
    print(f"Input: {features}")
    print(f"Predicted PM Temp: {prediction[0]}")
    print(f"Actual PM Temp (from CSV): 87.96")
