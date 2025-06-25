import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
import random
import pickle
from tqdm import tqdm


def main():
    ds_full = pd.read_pickle('pd_splits/dist_emb_msc.pkl')
    #print(ds_full.columns)
    ds_full = ds_full[ds_full['IntensityOfNumberBills'] != 1]

    X, y = ds_full.drop(['IntensityOfNumberBills'], axis=1), ds_full['IntensityOfNumberBills']
    #X.drop(['AverageBill', 'TruncatedAverageBill'], axis=1, inplace=True)
    #print(y.iloc[:10])
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.15, random_state=59)
    print(len(X_train), len(X_test), len(y_train), len(y_test))

    scaler = StandardScaler()
    numeric_cols = [col for col in X_train.columns if (X_train[col].dtype in ['int64', 'float64', 'int32', 'float32'])]

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # print(np.array([X[col].to_numpy() for col in X.columns if 'emb' in col]).shape)
    # embs_values = np.array([X[col].to_numpy() for col in X.columns if 'emb' in col])
    # print(min(embs_values.min(axis=1)), max(embs_values.max(axis=1)))

    model = CatBoostClassifier()
    model.fit(X_train, y_train, verbose=100)

    preds = model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, preds, multi_class='ovr')
    acc = accuracy_score(y_test, model.predict(X_test))

    print('Catboost Intensity WITH aggregated embeds')
    print("Roc Auc Score:", roc_auc)
    print("Accuracy Score:", acc)

    clf = TabPFNClassifier()
    clf.fit(X_train, y_train)

    preds = clf.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, preds, multi_class='ovr',)
    acc = accuracy_score(y_test, clf.predict(X_test))

    print('Tabpfn Intensity WITH aggregated embeds')
    print("Roc Auc Score:", roc_auc)
    print("Accuracy Score:", acc)

    X, y = ds_full.drop(['IntensityOfNumberBills'], axis=1), ds_full['IntensityOfNumberBills']
    #X.drop(['AverageBill', 'TruncatedAverageBill'], axis=1, inplace=True)
    columns_to_drop = [col for col in X.columns if 'emb' in col]
    X.drop(columns_to_drop, axis=1, inplace=True)
    # print(y.iloc[:10])
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.15, random_state=59)
    print(len(X_train), len(X_test), len(y_train), len(y_test))

    scaler = StandardScaler()
    numeric_cols = [col for col in X_train.columns if (X_train[col].dtype in ['int64', 'float64', 'int32', 'float32'])]

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    model = CatBoostClassifier()
    model.fit(X_train, y_train, verbose=100)

    preds = model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, preds, multi_class='ovr')
    acc = accuracy_score(y_test, model.predict(X_test))

    print('Catboost Intensity WITHOUT aggregated embeds')
    print("Roc Auc Score:", roc_auc)
    print("Accuracy Score:", acc)

    clf = TabPFNClassifier()
    clf.fit(X_train, y_train)

    preds = clf.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, preds, multi_class='ovr')
    acc = accuracy_score(y_test, clf.predict(X_test))

    print('Tabpfn Intensity WITHOUT aggregated embeds')
    print("Roc Auc Score:", roc_auc)
    print("Accuracy Score:", acc)


if __name__ == '__main__':
    main()
