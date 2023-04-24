import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from bayes_opt import BayesianOptimization

# Load the ICU patient data and select relevant features
data = pd.read_csv("icu_patient_data.csv")
# Choose relevant features
relevant_features = ["age", "gender", "heart_rate", "respiratory_rate", "temperature", "systolic_bp", "diastolic_bp", "mean_bp", "spo2"]
X = data[relevant_features]
y = data["sepsis"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter search space for XGBoost
param_space = {
    "max_depth": (3, 10),
    "gamma": (0, 1),
    "subsample": (0.5, 1),
    "colsample_bytree": (0.5, 1),
    "learning_rate": (0.01, 0.1),
    "n_estimators": (50, 1000)
}

# Define the objective function to optimize XGBoost hyperparameters using Bayesian optimization
def objective_function(max_depth, gamma, subsample, colsample_bytree, learning_rate, n_estimators):
    # Define the XGBoost model with the given hyperparameters
    xgb_model = xgb.XGBClassifier(max_depth=int(max_depth),
                                  gamma=gamma,
                                  subsample=subsample,
                                  colsample_bytree=colsample_bytree,
                                  learning_rate=learning_rate,
                                  n_estimators=int(n_estimators),
                                  random_state=42)
    # Train the model and make predictions on the testing set
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    # Calculate and return the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Use Bayesian optimization to find the best hyperparameters for XGBoost
optimizer = BayesianOptimization(f=objective_function, pbounds=param_space, random_state=42)
optimizer.maximize(init_points=5, n_iter=20)

# Define a function that takes patient data as input and returns an outcome (sepsis or not)
def predict_sepsis(patient_data):
    # Preprocess the patient data (e.g., one-hot encoding, scaling) and select relevant features
    patient_data = pd.DataFrame(patient_data, columns=relevant_features)
    patient_data = pd.get_dummies(patient_data)
    patient_data = (patient_data - X.mean()) / X.std()
    patient_data = patient_data.fillna(0) # Replace missing values with 0
    patient_data = patient_data[X.columns] # Keep only relevant features
    # Load the best hyperparameters found by Bayesian optimization and train an XGBoost model
    best_params = optimizer.max["params"]
    xgb_model = xgb.XGBClassifier(max_depth=int(best_params["max_depth"]),
                                  gamma=best_params["gamma"],
                                  subsample=best_params["subsample"],
                                  colsample_bytree=best_params["colsample_bytree"],
                                  learning_rate=best_params["learning_rate"],
                                  n_estimators=int(best_params["n_estimators"]),
                                  random_state=42)
    xgb_model.fit(X, y)
    # Make predictions on the patient data and return the outcome (sepsis or not)
    prediction = xgb_model.predict(patient_data)
    return "Sepsis"
