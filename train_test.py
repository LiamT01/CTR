"""Trains and tests the model"""

import pickle

import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from model_config import args

if __name__ == '__main__':
    data = pd.read_csv(args.features)
    # Cross-validation for 100 times
    spliter = ms.ShuffleSplit(n_splits=100, test_size=0.2, random_state=args.random_state)
    RMSE, MSE, PEARSON = [], [], []

    counter = 1
    with tqdm(total=100) as bar:
        for train, test in spliter.split(data):
            x_train, y_train, x_test, y_test = data.iloc[train, :-1], data.iloc[train, -1], \
                                               data.iloc[test, :-1], data.iloc[test, -1]

            cluster = KMeans(n_clusters=2, random_state=args.random_state).fit(x_train.values)
            labels_train = cluster.labels_
            labels_test = cluster.predict(x_test.values)

            reg1 = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse',
                                    n_estimators=args.n_estimators, max_depth=args.max_depth,
                                    min_child_weight=args.min_child_weight, subsample=args.subsample,
                                    gamma=args.gamma, colsample_bytree=args.colsample_bytree,
                                    learning_rate=args.learning_rate, reg_alpha=args.reg_alpha,
                                    reg_lambda=args.reg_lambda, early_stopping_rounds=args.early_stopping_rounds)

            reg2 = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse',
                                    n_estimators=args.n_estimators, max_depth=args.max_depth,
                                    min_child_weight=args.min_child_weight, subsample=args.subsample,
                                    gamma=args.gamma, colsample_bytree=args.colsample_bytree,
                                    learning_rate=args.learning_rate, reg_alpha=args.reg_alpha,
                                    reg_lambda=args.reg_lambda, early_stopping_rounds=args.early_stopping_rounds)

            eval_set1 = [(x_test.iloc[labels_test == 1], y_test.iloc[labels_test == 1])]
            eval_set2 = [(x_test.iloc[labels_test == 0], y_test.iloc[labels_test == 0])]

            reg1.fit(x_train.iloc[labels_train == 1], y_train.iloc[labels_train == 1],
                     eval_set=eval_set1, verbose=False)
            reg2.fit(x_train.iloc[labels_train == 0], y_train.iloc[labels_train == 0],
                     eval_set=eval_set2, verbose=False)

            preds = np.concatenate([reg1.predict(x_test.iloc[labels_test == 1]),
                                    reg2.predict(x_test.iloc[labels_test == 0])])

            gt = np.concatenate([y_test.iloc[labels_test == 1].to_numpy(),
                                 y_test.iloc[labels_test == 0].to_numpy()])

            mse = mean_squared_error(gt, preds)
            rmse = np.sqrt(mse)
            pearson = pearsonr(gt, preds)[0]

            RMSE.append(rmse)
            MSE.append(mse)
            PEARSON.append(pearson)

            print(f'rmse={rmse:.4f}, mse={mse:.4f}, pearsonr={pearson:.4f}')

            pickle.dump(reg1, open(f'weights/reg1/reg1_{counter:03d}', 'wb'))
            pickle.dump(reg2, open(f'weights/reg2/reg2_{counter:03d}', 'wb'))
            pickle.dump(cluster, open(f'weights/cluster/cluster_{counter:03d}', 'wb'))
            counter += 1

            bar.update(1)

    avg_rmse = sum(RMSE) / len(RMSE)
    avg_mse = sum(MSE) / len(MSE)
    avg_r = sum(PEARSON) / len(PEARSON)

    print(f'Average: rmse={avg_rmse:.4f}, mse={avg_mse:.4f}, pearsonr={avg_r:.4f}')
