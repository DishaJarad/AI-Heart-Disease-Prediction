import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from scipy import stats


df = pd.read_csv("Dataset.csv")


selected_features = ["age", "cp", "thalach", "oldpeak", "ca", "thal", "exang"]
X = df[selected_features]
y = df["target"]


z_scores = np.abs(stats.zscore(X))
X = X[(z_scores < 3).all(axis=1)]
y = y.loc[X.index]  


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

def choose_model():
    print("\nChoose Algorithm:")
    print("1: AdaBoost")
    print("2: XGBoost")

    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.3, 0.5]
    }
        base_model = AdaBoostClassifier(random_state=42)
        model_name = "AdaBoost"
    
    elif choice == "2":
        param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.3],
        'max_depth': [3, 5, 7]
    }
        base_model = XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)
        model_name = "XGBoost"
        
    else:
        print("Invalid choice! Defaulting to AdaBoost.")
        base_model = AdaBoostClassifier(random_state=42)
        model_name = "AdaBoost"
        param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.3, 0.5]
        }

    grid_search = RandomizedSearchCV(base_model, param_grid, n_iter=10, cv=5, n_jobs=-1, random_state=42)
    grid_search.fit(X_train_scaled, y_train)

    print(f"\nBest Parameters for {model_name}: {grid_search.best_params_}")

    return grid_search.best_estimator_, model_name


best_model, model_name = choose_model()


cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)
print(f"\n{model_name} Cross-Validation Scores:")
print(f"Mean: {cv_scores.mean():.3f}, Std: {cv_scores.std():.3f}")


val_pred = best_model.predict(X_val_scaled)
print(f"\nValidation Accuracy: {accuracy_score(y_val, val_pred):.3f}")
print("Validation Report:\n", classification_report(y_val, val_pred))


test_pred = best_model.predict(X_test_scaled)
print(f"\nTest Accuracy: {accuracy_score(y_test, test_pred):.3f}")
print("Test Report:\n", classification_report(y_test, test_pred))

def get_user_input():
    print("\nEnter patient details:")
    inputs = []

    inputs.append(int(input("Age: ")))
    inputs.append(int(input("Chest Pain Type (0-3): ")))
    inputs.append(int(input("Max Heart Rate Achieved: ")))
    inputs.append(float(input("ST Depression Induced by Exercise: ")))
    inputs.append(int(input("Number of Major Vessels (0-4): ")))
    inputs.append(int(input("Thalassemia Type (0-3): ")))
    inputs.append(int(input("Exercise-Induced Angina (1 = Yes, 0 = No): ")))

    return np.array([inputs])


user_data = get_user_input()
user_df = pd.DataFrame(user_data, columns=selected_features)
user_scaled = scaler.transform(user_df)
prediction = best_model.predict(user_scaled)
result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
print("\nPrediction Result:", result)

