import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import traceback

log_file = open("execution_log.txt", "w")

def log(msg):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

try:
    log("Starting script...")
    
    # Load Data
    try:
        df = pd.read_csv('pmsm_temperature_data.csv')
        log(f"Data loaded: {df.shape}")
    except FileNotFoundError:
        log("Error: pmsm_temperature_data.csv not found.")
        exit(1)

    # Preprocessing
    target = 'pm'
    # features = ['ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'torque', 'i_d', 'i_q']
    # Removed torque as per analysis
    features = ['ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'i_d', 'i_q']

    X = df[features]
    y = df[target]

    # Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log("Data split.")

    # Scaling
    # Using MinMaxScaler as per latest requirement
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    log("Data scaled using MinMaxScaler.")

    # Model Building & Evaluation
    # Based on notebook analysis, Decision Tree is selected.
    # Linear Regression and Random Forest were evaluated but DT selected.
    
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train_scaled, y_train)
    log("Decision Tree Model trained.")

    # Evaluation
    y_pred = model.predict(X_test_scaled)
    from sklearn.metrics import mean_squared_error, r2_score
    import math
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    log(f"Model Evaluated. RMSE: {rmse:.4f}, R2: {r2:.4f}")


    # Save Model and Scaler
    with open('model.save', 'wb') as f:
        pickle.dump(model, f)
    
    with open('transform.save', 'wb') as f:
        pickle.dump(scaler, f)

    # Also save into Flask folder as requested by structure
    with open('Flask/model.save', 'wb') as f:
        pickle.dump(model, f)

    with open('Flask/transform.save', 'wb') as f:
        pickle.dump(scaler, f)

    log("Model and scaler saved successfully to Project Folder and Flask subfolder.")

except Exception as e:
    log(f"An error occurred: {str(e)}")
    traceback.print_exc(file=log_file)
finally:
    log_file.close()
