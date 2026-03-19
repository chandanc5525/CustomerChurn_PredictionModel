from src.data_ingestion import data_ingestion 
from src.data_preprocessing import data_preprocessing
from src.model_building import model_building 
def main():
    
    # Step1: Data Ingestion
    df = data_ingestion()
    print(df.shape)
    
    # Step2: Data Preprocessing
    X_train,X_test,y_train,y_test,preprocessor,svd = data_preprocessing(df)
    print(X_train.shape)
    print(X_test.shape)
    
    # Step3: Model Building
    
    model = model_building(X_train, X_test, y_train, y_test,preprocessor,svd)

    print(model)
    
    
main()