import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load preprocessed data
Xr_train, Xr_test, yr_train, yr_test = joblib.load('models/regression_data.pkl')

# Build Random Forest Regressor
print("🔄 Training Regression Model...")
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(Xr_train, yr_train)

# Predict
yr_pred = reg.predict(Xr_test)

# Results
mse  = mean_squared_error(yr_test, yr_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(yr_test, yr_pred)
r2   = r2_score(yr_test, yr_pred)

print("\n📊 Regression Results:")
print(f"✅ R2 Score  : {r2:.4f}")
print(f"✅ RMSE     : {rmse:.4f}")
print(f"✅ MAE      : {mae:.4f}")

# Save model
joblib.dump(reg, 'models/regressor.pkl')
print("\n✅ Regression model saved!")