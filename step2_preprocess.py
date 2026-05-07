import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Create models folder
os.makedirs('models', exist_ok=True)

# Load data
df = pd.read_csv('irrigation_prediction.csv')

# Encode categorical columns
cat_cols = ['Soil_Type','Crop_Type','Crop_Growth_Stage','Season',
            'Irrigation_Type','Water_Source','Mulching_Used','Region']

le_dict = {}
df_enc = df.copy()

for col in cat_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Encode target column
le_target = LabelEncoder()
df_enc['Irrigation_Need'] = le_target.fit_transform(df['Irrigation_Need'])

print("✅ Classes:", le_target.classes_)

# Save encoders
joblib.dump(le_dict, 'models/label_encoders.pkl')
joblib.dump(le_target, 'models/target_encoder.pkl')

# Features and targets
X = df_enc.drop(columns=['Irrigation_Need'])
y_class = df_enc['Irrigation_Need']
y_reg   = df['Previous_Irrigation_mm']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'models/scaler.pkl')

# Split data
X_train, X_test, yc_train, yc_test = train_test_split(
    X_scaled, y_class, test_size=0.2, random_state=42)

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_scaled, y_reg, test_size=0.2, random_state=42)

# Save splits
joblib.dump((X_train, X_test, yc_train, yc_test), 'models/classification_data.pkl')
joblib.dump((Xr_train, Xr_test, yr_train, yr_test), 'models/regression_data.pkl')
joblib.dump(list(X.columns), 'models/feature_names.pkl')

print("✅ Preprocessing done!")
print("✅ Train size:", X_train.shape)
print("✅ Test size:", X_test.shape)
print("✅ All files saved in models/ folder")