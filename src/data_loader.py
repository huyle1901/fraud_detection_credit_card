import pandas as pd

class DataLoader:

    def __init__(self,file_path):
        self.file_path = file_path

    def load_data(self):
        # Read data from file_path
        try: 
            self.data = pd.read_csv(self.file_path)
            print("Data successfully loaded.")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def processed_data(self):
        if self.data is None:
            print("No data to preprocess. Please load the data first.")
            return None
        
        # Drop unnecessary columns
        columns_to_drop = ['first', 'last', 'street','trans_day_of_week']
        self.data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        print("Data preprocessing completed.")

        return self.data


    
