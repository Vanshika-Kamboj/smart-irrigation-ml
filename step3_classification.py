import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load preprocessed data
X_train, X_test, y_train, y_test = joblib.load('models/classification_data.pkl')
le_target = joblib.load('models/target_encoder.pkl')

# Build Random Forest Classifier
print("🔄 Training Classification Model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Results
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc * 100:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# Save model
joblib.dump(clf, 'models/classifier.pkl')
print("\n✅ Classification model saved!")