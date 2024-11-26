import pandas as pd
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader import DataLoader
from feature_engineering import FeatureEngineering
from fraud_detection_model import FraudDetectionModel


def main():

    # Step 1: Load data
    file_path = 'C:/Users/Admin/Desktop/fraud_detection_project/data/processed/clean_credit_card_transactions.csv'
    data_loader = DataLoader(file_path)
    data = data_loader.load_data()
    data = data_loader.processed_data()

    # Step 2: Feature engineering
    
    feature_engineer = FeatureEngineering(data)

    # Label Encoding
    label_cols = ['merchant', 'category', 'gender', 'state', 'city', 'job', 'merch_zipcode', 'age_group']
    data = feature_engineer.label_encode_columns(label_cols)

    # Scale columns
    data = feature_engineer.scale_columns()

    # Save encoders and scaler
    feature_engineer.save_encoders_and_scaler()

    # Save processed data
    processed_file = "C:/Users/Admin/Desktop/fraud_detection_project/data/features/features_credit_card_transactions.csv"
    data.to_csv(processed_file, index=False)
    print(f"Processed data saved to {processed_file}.")


    # Step 3: Train and Evaluate models

    print("\n--- Starting model training ---")
    model_path = "C:/Users/Admin/Desktop/fraud_detection_project/model"

    fraud_model = FraudDetectionModel(processed_file,model_path)
    
    fraud_model.load_data()
    fraud_model.split_data()

    models = {
        "Gradient Boosting": GradientBoostingClassifier(),
    }

    fraud_model.train_and_evalute_model(models)

if __name__ == "__main__":
    main()