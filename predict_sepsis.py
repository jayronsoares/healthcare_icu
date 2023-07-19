import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization

class SepsisPredictor:
    def __init__(self):
        self.selected_features = ["age", "gender", "heart_rate", "respiratory_rate", "systolic_bp", "diastolic_bp", "mean_bp", "spo2", "temperature", "urine_output", "wbc_count", "platelet_count", "glucose", "sodium", "potassium", "creatinine", "bun", "lactate", "albumin", "bnp", "pao2", "pco2", "ph", "bicarbonate", "blood_culture", "urine_culture", "ventilator", "central_line", "urinary_catheter"]
        self.le = LabelEncoder()
        self.xgb_model = None

    def preprocess_data(self, data):
        X = data[self.selected_features]
        y = data["sepsis"]
        X["gender"] = self.le.fit_transform(X["gender"])
        X["ventilator"] = self.le.fit_transform(X["ventilator"])
        X["central_line"] = self.le.fit_transform(X["central_line"])
        X["urinary_catheter"] = self.le.fit_transform(X["urinary_catheter"])
        X["blood_culture"] = X["blood_culture"].astype(int)
        X["urine_culture"] = X["urine_culture"].astype(int)
        return X, y

    def train_model(self, X, y):
        def xgb_cv(max_depth, learning_rate, n_estimators, gamma, min_child_weight, max_delta_step, subsample, colsample_bytree):
            model = XGBClassifier(max_depth=int(max_depth),
                                  learning_rate=learning_rate,
                                  n_estimators=int(n_estimators),
                                  gamma=gamma,
                                  min_child_weight=min_child_weight,
                                  max_delta_step=int(max_delta_step),
                                  subsample=subsample,
                                  colsample_bytree=colsample_bytree,
                                  random_state=42,
                                  tree_method='gpu_hist')
            cv_result = XGBClassifier.cross_val_score(model, X, y, cv=5).mean()
            return cv_result

        xgbBO = BayesianOptimization(f=xgb_cv, pbounds={'max_depth': (3, 20), 'learning_rate': (0.01, 0.3), 'n_estimators': (100, 1000), 'gamma': (0, 1), 'min_child_weight': (1, 10), 'max_delta_step': (0, 10), 'subsample': (0.5, 1), 'colsample_bytree': (0.5, 1)})
        xgbBO.maximize(n_iter=10, init_points=10)

        # Get the optimal hyperparameters
        params = xgbBO.max['params']
        params['max_depth'] = int(params['max_depth'])
        params['n_estimators'] = int(params['n_estimators'])
        params['max_delta_step'] = int(params['max_delta_step'])

        # Train the final model with the optimal hyperparameters
        self.xgb_model = XGBClassifier(**params, tree_method='gpu_hist')
        self.xgb_model.fit(X, y)

    def predict_sepsis(self, patient_data):
        patient_data = pd.DataFrame(patient_data, index=[0], columns=self.selected_features)
        patient_data["gender"] = self.le.transform(patient_data["gender"])
        patient_data["ventilator"] = self.le.transform(patient_data["ventilator"])
        patient_data["central_line"] = self.le.transform(patient_data["central_line"])
        patient_data["urinary_catheter"] = self.le.transform(patient_data["urinary_catheter"])
        patient_data["blood_culture"] = patient_data["blood_culture"].astype(int)
        patient_data["urine_culture"] = patient_data["urine_culture"].astype(int)
        pred = self.xgb_model.predict(patient_data)[0]
        if pred == 1:
            return "Sepsis detected."
        else:
            return "Sepsis not detected."


# Example usage
if __name__ == "__main__":
    # 1. Data Collection and Preprocessing
    data = pd.read_csv("icu_patient_data.csv")
    data = data.dropna()

    predictor = SepsisPredictor()
    X, y = predictor.preprocess_data(data)

    # 2. Model Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    predictor.train_model(X_train, y_train)

    # 3. Model Deployment
    patient_data = {
        "age": 60,
        "gender": "Female",
        "heart_rate": 80,
        "respiratory_rate": 18,
        "systolic_bp": 120,
        # Include other relevant patient data based on ICU standard protocols
        "urine_output": 1000,
        "wbc_count": 12000,
        "platelet_count": 250000,
        "glucose": 120,
        "sodium": 140,
        "potassium": 4.5,
        "creatinine": 0.9,
        "bun": 20,
        "lactate": 2.0,
        "albumin": 3.5,
        "bnp": 100,
        "pao2": 80,
        "pco2": 40,
        "ph": 7.4,
        "bicarbonate": 25,
        "blood_culture": 1,
        "urine_culture": 1,
        "ventilator": "Yes",
        "central_line": "No",
        "urinary_catheter": "Yes"
    }

    outcome = predictor.predict_sepsis(patient_data)
    print(outcome)
