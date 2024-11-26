import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score,r2_score,precision_score,recall_score
import joblib
import os

class FraudDetectionModel:
    def __init__(self,data_path,model_save_dir):
        self.data_path = data_path
        self.model_save_dir = model_save_dir
        self.models = None
        self.results = {}
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def load_data(self):

        data = pd.read_csv(self.data_path)
        self.X = data.drop(columns=['is_fraud'])  # Các cột đặc trưng
        self.y = data['is_fraud']  # Cột mục tiêu
        print("Data loaded successfully.")

    def split_data(self, test_size = 0.3, random_state =42):
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        print("Data split completed.")

    def train_and_evalute_model(self, models):

        self.models = models
        for model_name, model in models.items():
            print(f"\nTraining Model: {model_name}")
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)

           # Check if model supports probability predictions
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(self.X_test)[:, 1]
            roc_auc = roc_auc_score(self.y_test, y_prob)
        else:
            y_prob = None
            roc_auc = "Not applicable (model does not support probability estimates)"

        # Evaluate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        report = classification_report(self.y_test, y_pred)

        # Print results
        print(f"\n{model_name} Classification Report:")
        print(report)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"ROC-AUC Score: {roc_auc}\n")
    
        # Save the model to a file
        model_file_path = os.path.join(self.model_save_dir, f"{model_name.replace(" ","_")}.pkl")
        joblib.dump(model, model_file_path)
        print(f"Model '{model_name}' saved to {model_file_path}.")

    