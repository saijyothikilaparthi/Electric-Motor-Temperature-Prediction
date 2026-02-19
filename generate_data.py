import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# specific columns for Electric Motor Temperature
# commonly used features: ambient, coolant, u_d, u_q, motor_speed, torque, i_d, i_q
# target: pm (permanent magnet temperature), stator_yoke, stator_tooth, stator_winding

n_samples = 1000

data = {
    'ambient': np.random.normal(25, 5, n_samples),
    'coolant': np.random.normal(50, 10, n_samples),
    'u_d': np.random.normal(0, 50, n_samples),
    'u_q': np.random.normal(50, 30, n_samples),
    'motor_speed': np.random.normal(3000, 1000, n_samples),
    'torque': np.random.normal(50, 20, n_samples),
    'i_d': np.random.normal(-100, 30, n_samples),
    'i_q': np.random.normal(100, 30, n_samples),
    'pm': np.random.normal(60, 15, n_samples), # Target
    'stator_yoke': np.random.normal(55, 12, n_samples),
    'stator_tooth': np.random.normal(58, 13, n_samples),
    'stator_winding': np.random.normal(65, 15, n_samples),
    'profile_id': np.random.randint(1, 20, n_samples)
}

df = pd.DataFrame(data)

# Introduce some correlations for realism (simple linear relationships + noise)
df['pm'] = 20 + 0.5 * df['ambient'] + 0.3 * df['coolant'] + 0.01 * df['motor_speed'] + 0.1 * df['i_q'] + np.random.normal(0, 2, n_samples)

# Save to CSV
df.to_csv(r'c:\Users\ASUS\Desktop\saijyothi\Project Folder\mpsm_temperature_data.csv', index=False)
print("Synthetic data generated successfully.")
