import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor


def main():
    to_take_train_rows = 10000
    X_train = pd.read_pickle('pd_splits/x_train_msk_merged.pkl')[:to_take_train_rows]
    X_test = pd.read_pickle('pd_splits/x_test_msk_merged.pkl')
    y_train = pd.read_pickle('pd_splits/y_train_msk_merged.pkl').to_numpy()[:to_take_train_rows]
    y_test = pd.read_pickle('pd_splits/y_test_msk_merged.pkl').to_numpy()

    print(len(X_train), len(X_test), len(y_train), len(y_test))

    model = CatBoostRegressor(ignored_features=['index', 'level_0'])
    model.fit(X_train, y_train, verbose=100)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print('Catboost NO embeds')
    print("Root Mean Squared Error (RMSE):", mse ** 0.5)
    print("R² Score:", r2)

    X_train = pd.read_pickle('X_train_mosc_wembs.pickle')[:to_take_train_rows]
    X_test = pd.read_pickle('X_test_mosc_wembs.pickle')
    y_train = pd.read_pickle('pd_splits/y_train_msk_merged.pkl').to_numpy()[:to_take_train_rows]
    y_test = pd.read_pickle('pd_splits/y_test_msk_merged.pkl').to_numpy()

    model = CatBoostRegressor(ignored_features=['index', 'level_0'])
    model.fit(X_train, y_train, verbose=100)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print('Catboost WITH embeds')
    print("Root Mean Squared Error (RMSE):", mse ** 0.5)
    print("R² Score:", r2)

    X_train = pd.read_pickle('X_train_mosc_wembs.pickle')[:to_take_train_rows]
    X_test = pd.read_pickle('X_test_mosc_wembs.pickle')
    y_train = pd.read_pickle('pd_splits/y_train_msk_merged.pkl').to_numpy()[:to_take_train_rows]
    y_test = pd.read_pickle('pd_splits/y_test_msk_merged.pkl').to_numpy()

    columns_to_drop = [col for col in X_train.columns if 'emb' not in col]
    model = CatBoostRegressor(ignored_features=columns_to_drop)
    model.fit(X_train, y_train, verbose=100)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print('Catboost ONLY embeds')
    print("Root Mean Squared Error (RMSE):", mse ** 0.5)
    print("R² Score:", r2)


if __name__ == '__main__':
    main()
