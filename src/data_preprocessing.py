from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np


def data_preprocessing(df):

    df = df.drop_duplicates()

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    X = df.drop(columns=['Churn', 'customerID'], errors='ignore')
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=1,
        stratify=y
    )

    numerical_col = X_train.select_dtypes(exclude='object').columns
    categorical_col = X_train.select_dtypes(include='object').columns

    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_col),
        ('cat', categorical_pipeline, categorical_col)
    ])

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    sm = SMOTE(sampling_strategy=0.5, random_state=1)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    # Step 1: Fit SVD with higher components
    svd_temp = TruncatedSVD(n_components=100, random_state=1)
    X_train_temp = svd_temp.fit_transform(X_train)

    # Step 2: Calculate cumulative variance
    cumulative_variance = np.cumsum(svd_temp.explained_variance_ratio_)

    # Step 3: Find components for 95% variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

    # Step 4: Final SVD
    svd = TruncatedSVD(n_components=n_components_95, random_state=1)
    X_train = svd.fit_transform(X_train)
    X_test = svd.transform(X_test)

    print(f"Selected Components for 95% variance: {n_components_95}")

    return X_train, X_test, y_train, y_test, preprocessor,svd