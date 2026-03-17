from src.data_ingestion import data_ingestion 


def main():
    
    # Step1: Data Ingestion
    df = data_ingestion()
    print(df.shape)
    
main()