
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_scale_data(path):
    df = pd.read_csv(path)
    features = df.drop("quality", axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, df
