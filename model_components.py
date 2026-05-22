import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BatteryFeatureEngineer(BaseEstimator, TransformerMixin):
    """Membentuk fitur turunan dari ringkasan sensor."""

    def fit(self, X, y=None):
        self.input_features_ = list(X.columns)
        transformed = self.transform(X)
        self.output_features_ = list(transformed.columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=getattr(self, "input_features_", X.columns)).copy()

        X["voltage_range"] = X["voltage_max"] - X["voltage_min"]
        X["current_range"] = X["current_max"] - X["current_min"]
        X["temperature_range"] = X["temperature_max"] - X["temperature_min"]
        X["voltage_stability_ratio"] = X["voltage_std"] / (X["voltage_mean"].abs() + 1e-6)
        X["temperature_stability_ratio"] = X["temperature_std"] / (
            X["temperature_mean"].abs() + 1e-6
        )
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.output_features_)


class IQRClipper(BaseEstimator, TransformerMixin):
    """Membatasi nilai ekstrem dengan metode IQR."""

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.clip(self.lower_, self.upper_, axis=1)
