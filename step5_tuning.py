import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Load data
X_train, X_test, y_train, y_test = joblib.load('models/classification_data.pkl')

# Parameter grid to search
param_grid = {
    'n_estimators'     : [50, 100, 200],
    'max_depth'        : [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4],
    'max_features'     : ['sqrt', 'log2']
}

print("🔄 Tuning Random Forest... (takes 2-3 minutes)")

# RandomizedSearchCV
rf = RandomForestClassifier(random_state=42)
search = RandomizedSearchCV(
    rf,
    param_distributions=param_grid,
    n_iter=20,
    cv=3,
    scoring='accuracy',
    random_state=42,
    verbose=1
)

search.fit(X_train, y_train)

# Best results
print("\n✅ Best Parameters:")
print(search.best_params_)
print(f"\n✅ Best CV Accuracy: {search.best_score_ * 100:.2f}%")

# Test accuracy
best_model = search.best_estimator_
test_acc = best_model.score(X_test, y_test)
print(f"✅ Test Accuracy   : {test_acc * 100:.2f}%")

# Save best model
joblib.dump(best_model, 'models/best_classifier.pkl')
print("\n✅ Best tuned model saved!")