import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set style
sns.set(style="whitegrid")

# Load data
df = pd.read_csv('pmsm_temperature_data.csv')

# Ensure output directory exists
output_dir = 'Flask/static/images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(f'{output_dir}/correlation_heatmap.png')
plt.close()

# 2. PM Temperature Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['pm'], kde=True, color='blue')
plt.title('Distribution of Rotor Temperature (PM)')
plt.xlabel('Temperature')
plt.savefig(f'{output_dir}/pm_distribution.png')
plt.close()

# 3. Ambient vs PM Temperature
plt.figure(figsize=(10, 6))
sns.scatterplot(x='ambient', y='pm', data=df, alpha=0.5)
plt.title('Ambient Temperature vs Rotor Temperature')
plt.savefig(f'{output_dir}/ambient_vs_pm.png')
plt.close()

# 4. Coolant vs PM Temperature
plt.figure(figsize=(10, 6))
sns.scatterplot(x='coolant', y='pm', data=df, alpha=0.5, color='green')
plt.title('Coolant Temperature vs Rotor Temperature')
plt.savefig(f'{output_dir}/coolant_vs_pm.png')
plt.close()

print("Plots generated successfully in Flask/static/images")
