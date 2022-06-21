"""Configuration for train_test.py"""

from utils import AttrDict


args = AttrDict({
    "features": "data/features.csv",  # Path to the .csv file containing features
    "random_state": 0,  # Random state for reproducibility
    # Below are parameters for XGBoost
    "n_estimators": 2000,
    "max_depth": 5,
    "min_child_weight": 8,
    "subsample": 0.95,
    "gamma": 0.6,
    "colsample_bytree": 0.2,
    "learning_rate": 0.05,
    "reg_alpha": 1.5,
    "reg_lambda": 3,
    "early_stopping_rounds": 30  # Stop training if there has been no improvement for 30 rounds
})
