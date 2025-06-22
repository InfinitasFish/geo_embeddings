import pandas as pd

if __name__ == '__main__':
    df = pd.read_pickle('X_train_mosc_wembs.pickle')
    print(len(df), df.columns)