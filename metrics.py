from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


def compute_metrics(true, pred):
    return {"RMSE": np.sqrt(mean_squared_error(true, pred)),
            "MAE": MAE(true, pred),
            "MAPE": MAPE(true, pred)}


def MAE(true, pred):
    return mean_absolute_error(true, pred)


def MAPE(true, pred, eps=1e-8):
    return np.mean(np.abs((true - pred) / (true + eps))) * 100
