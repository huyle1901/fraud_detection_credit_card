import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


class FeatureEngineering:
    def __init__(self, data):
        self.data = data
        self.encoders = {}
        self.scaler = None

    def label_encode_columns(self, columns):
        for col in columns:
            encoder = LabelEncoder()
            self.data[col] = encoder.fit_transform(self.data[col])
            self.encoders[col] = encoder
        print("Label encoding completed.")
        return self.data
    
    def scale_columns(self, columns=None):

        self.scaler = StandardScaler()

        # Nếu không truyền danh sách cột, chọn tất cả cột trừ 'is_fraud'
        if columns is None:
            columns = [col for col in self.data.columns if col != 'is_fraud']

        
        numeric_columns = self.data[columns].select_dtypes(include=['int64', 'float64']).columns
       
        self.data[numeric_columns] = self.scaler.fit_transform(self.data[numeric_columns])
        
        print("Scaling completed for all numeric columns except 'is_fraud'.")
        return self.data
    
    def save_encoders_and_scaler(self, encoder_path='encoders.pkl',scaler_path='scaler.pkl'):
        joblib.dump(self.encoders,encoder_path)
        joblib.dump(self.scaler,scaler_path)
        print(f"Encoders saved to {encoder_path}.")
        print(f"Scaler saved to {scaler_path}.")

