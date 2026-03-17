import os 
import pandas as pd 

# Path Configuration:
data_path = os.path.join("data", "churn.csv")  
model_dir = os.path.join("models")            
os.makedirs(model_dir, exist_ok=True)
pickle_path = os.path.join(model_dir, "churn_model.pkl")

# Data Loading 
def data_ingestion():    
    # Dataloading and it will return Dataframe
    df = pd.read_csv(data_path)
    return df



