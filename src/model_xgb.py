from xgboost import XGBRegressor

def build_xgb_model():
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    return model
from xgboost import XGBRegressor

def build_xgb_model():
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    return model
