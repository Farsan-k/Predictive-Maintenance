import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def encode_type_column(df):
    df = pd.get_dummies(df, columns=['Type'], drop_first=True)
    return df


def scale_features(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data


preprocessing_pipeline = Pipeline([
    ('encoding', FunctionTransformer(encode_type_column)),
    ('scaling', FunctionTransformer(scale_features))
])