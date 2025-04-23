import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from catboost import CatBoostRegressor


def main():

    X_train = pd.read_pickle('pd_splits_0416/x_train.pkl')[:10000]
    X_test = pd.read_pickle('pd_splits_0416/x_test.pkl')
    y_train = pd.read_pickle('pd_splits_0416/y_train.pkl')[:10000]
    y_test = pd.read_pickle('pd_splits_0416/y_test.pkl')

    # Initialize the regressor
    regressor = CatBoostRegressor(loss_function='RMSE')
    print('Fit started')
    regressor.fit(X_train, y_train, verbose=100)
    print('Fit completed')

    # Predict on the test set
    print('Predictions started')
    predictions = regressor.predict(X_test)
    print('Predictions completed')

    # Evaluate
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Mean Squared Error (MSE):", mse)
    print("R² Score:", r2)


if __name__ == '__main__':
    main()
