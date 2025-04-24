from xgboost import XGBRegressor

def build_xgb_model():
    return XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
