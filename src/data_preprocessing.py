from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,LabelEncoder

''' 
Workflow Design:
1. Clean the unwanted columns from the Dataset
2. Split the data into X and y
3. Split the dataset into Train and Test
4. Split the data into numerical and categorical columns
5. Use Pipeline for Numerical and Categorical columns
6. Use Column Transformer to fit our model
7. Use Smote Technique and then PCA (For Dimension Reductionality)
8. Return X_train,X_test,y_train,y_test

'''
def data_preprocessing(df):
    
    # Step1: Clean Duplicated Values
    df = df.drop_duplicates()
    
    # Encode target to 0 and 1
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    
    # Step2: Split the Dataset into X and y
    X = df.drop(columns = ['customerID','Churn'])
    y = df['Churn']
    
    # Step3: Split the Dataset into Train and Test
    X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                     test_size = 0.3,
                                                     random_state = 1)
    
    # Step4: Segregate Data into Numerical and Categorical Columns
    numerical_col = X_train.select_dtypes(exclude = 'object').columns
    categorical_col = X_train.select_dtypes(include = 'object').columns
    
    # Step5: Using Numerical_Pipeline and Categorical_Pipeline
    
    Numerical_Pipeline = Pipeline(steps = [
        ('Imputer',SimpleImputer(strategy= 'median')),
        ('Scaling',MinMaxScaler())
    ])
    
    Categorical_Pipeline = Pipeline(steps=[
        ('Imputer',SimpleImputer(strategy= 'most_frequent')),
        ('Encoder',OneHotEncoder(handle_unknown= 'ignore'))
    ])
    
    Preprocessor = ColumnTransformer(transformers= [
        ('Numerical_pipe',Numerical_Pipeline,numerical_col),
        ('Categorical_pipe',Categorical_Pipeline,categorical_col)
    ])
    
    X_train = Preprocessor.fit_transform(X_train)
    X_test = Preprocessor.transform(X_test)
    
    # Use SMOTE Technique
    sm = SMOTE()
    
    X_train,y_train = sm.fit_resample(X_train,y_train)
    
    # Use PCA (Principal Component Analysis: Dimensional Reduction Technique)
    
    pca = PCA()
    
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    return X_train,X_test,y_train,y_test

    
    
    
    
    
    
    
    
    
    