import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor


def main():
    # Load data, tabpfn can't train (officially) on more than 10k examples
    X_train = pd.read_pickle('pd_splits_2404/x_train.pkl')[:10000]
    X_test = pd.read_pickle('pd_splits_2404/x_test.pkl')
    y_train = pd.read_pickle('pd_splits_2404/y_train.pkl')[:10000]
    y_test = pd.read_pickle('pd_splits_2404/y_test.pkl')

    print(len(X_train), len(y_train), len(X_test), len(y_test))

    # Initialize the regressor
    regressor = TabPFNRegressor()
    print('Fit started')
    regressor.fit(X_train, y_train)
    print('Fit completed')

    # Predict on the test set
    print('Predictions started')
    predictions = regressor.predict(X_test)
    print('Predictions completed')

    # Evaluate
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Root Mean Squared Error (RMSE):", mse**0.5)
    print("R² Score:", r2)


if __name__ == '__main__':
    main()
