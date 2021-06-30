import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

# NOTE: Make sure that the outcome column is labeled 'target' in the data file

# Average CV score on the training set was: -111.53314445928015
exported_pipeline = make_pipeline(
    RobustScaler(),
    XGBRegressor(learning_rate=0.5, max_depth=5, min_child_weight=20, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=1.0)
)

