import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('irrigation_prediction.csv')

# Explore
print("✅ Shape:", df.shape)
print("\n📋 Columns:", df.columns.tolist())
print("\n🔍 First 5 rows:")
print(df.head())
print("\n📊 Data Info:")
print(df.info())
print("\n🔢 Statistics:")
print(df.describe())
print("\n🎯 Irrigation Need Values:")
print(df['Irrigation_Need'].value_counts())