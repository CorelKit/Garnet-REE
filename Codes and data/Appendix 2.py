# -*- coding: utf-8 -*-
import copy
import os
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, make_scorer,
                             roc_auc_score)
from sklearn.metrics import (confusion_matrix, classification_report)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from skopt import BayesSearchCV
from skopt.space import Real, Integer


def detect_anomalies(row, columns_order):
    y_values = row.iloc[columns_order].values.astype(float)
    n = len(y_values)
    #  6 La, 7 Ce, 8 Pr, 9 Nd*, 10 Sm,
    #  11 Eu, 12 Gd, 13 Tb*, 14 Dy, 5 Y,
    #  15 Ho, 16 Er*, 17 Tm*, 18 Yb*, 19 Lu,
    lower = [1, 0.0001, 0.0001, 0.35, 0.0001,
             0.0001, 0.0001, 0.35, 0.0001, 0.0001,
             0.0001, 0.35, 0.35, 0.35, 1]
    upper = [1, 10000, 10000, 1.65, 10000,
             10000, 10000, 1.65, 10000, 10000,
             10000, 1.65, 1.65, 1.65, 1]
    for i in range(n):
        if i == 0:  # 首点用后邻值
            geo_mean = y_values[i]  # y_values[i + 1]
        elif i == n - 1:  # 末点用前邻值
            geo_mean = y_values[i]  # y_values[i - 1]
        else:  # 中间点用前后几何平均
            geo_mean = (y_values[i - 1] * y_values[i + 1]) ** 0.5
        ratio = y_values[i] / geo_mean
        if not (lower[i] <= ratio <= upper[i]):
            return True
    return False


def generate_metal_string(row, s, e):
    metals = ['W', 'Sn', 'Mo', 'Fe', 'Cu', 'Au', 'Pb', 'Zn']
    selected = []
    for idx in range(s, e):
        value = row[idx]
        if value != 0:
            selected.append(metals[idx - s])
    return '-'.join(selected)


warnings.filterwarnings("ignore")
current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_excel(os.path.join(current_dir, 'Appendix 1.xlsx'),
                   sheet_name="A1.Garnet REE ppm", engine='openpyxl')
print('Origin data:', df.shape)
test_results = []
stn, edn, mst, mend = 5, 20, 20, 28
thresholds = [0.001, 0.001, 0.001,  # W Sn Mo
              0.01, 1, 1,
              0.001, 0.001]

cols_to_binarize = df.columns[mst:mend]  # Label 0 1
for i, col in enumerate(cols_to_binarize):
    ser = pd.to_numeric(df[col], errors='coerce')
    df[col] = np.where(ser.isna() | (ser >= thresholds[i]), 1, 0)  # Label 0 1

# Del data with BDL
columns_to_check = df.columns[stn: edn]
df[columns_to_check] = df[columns_to_check].apply(pd.to_numeric, errors='coerce')
mask = (df[columns_to_check].gt(0).all(axis=1) & df[columns_to_check].notna().all(axis=1))
df = df[mask]
df = df.reset_index(drop=True)
print('Data without BDL:', df.shape)

# REE(garnet)/REE(chondrite_S_M_1989)
columns_to_process = df.columns[stn: edn]
Chondrite_S_M_1989 = [0.237, 0.612, 0.095, 0.467, 0.153,  # La Ce Pr Nd Sm
                      0.058, 0.2055, 0.0374, 0.254, 0.0566,  # Eu Gd Tb Dy Ho
                      0.1655, 0.0255, 0.17, 0.0254, 1.57, ]  # Er Tm Yb Lu Y
chondrite_series = pd.Series(Chondrite_S_M_1989, index=columns_to_process)
df[columns_to_process] = df[columns_to_process].div(chondrite_series)
columns_order1 = [5, 6, 7, 8, 9, 10, 11, 12, 13, 19, 14, 15, 16, 17, 18]
mask_clean = df.apply(lambda row: not detect_anomalies(row, columns_order1), axis=1)
df_clean = df[mask_clean]
df_clean = df_clean.reset_index(drop=True)
df_clean['La/Sm'] = df_clean['La'] / df_clean['Sm']
df_clean['δEu'] = df_clean['Eu'] / np.sqrt(df_clean['Sm'] * df_clean['Gd'])
df_clean['Y/Ho'] = df_clean['Y'] / df_clean['Ho']
df_clean['Ho/Yb'] = df_clean['Ho'] / df_clean['Yb']
new_order = ['Deposit', 'No.', 'Rock type', 'Minerals', 'Referances',
             'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb',
             'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Y',
             'La/Sm', 'δEu', 'Y/Ho', 'Ho/Yb',
             'W-deposit', 'Sn-deposit', 'Mo-deposit',
             'Fe-deposit', 'Cu-deposit', 'Au-deposit',
             'Pb-deposit', 'Zn-deposit',
             ]
df_clean = df_clean[new_order]
print('Cleaned data', df_clean.shape)

# save xlsx
with pd.ExcelWriter(os.path.join(current_dir, 'Cleaned_data.xlsx'), engine='xlsxwriter') as writer:
    df_clean.to_excel(writer, index=False, sheet_name='model')
print(os.path.join(current_dir, 'Cleaned_data.xlsx'))

# W 24, Sn 25, Mo 26, Fe 27, Cu 28, Au 29, Pb 30, Zn 31
gn, sdn, edn, stn, endn, sn_zhuchong = 0, 5, 24, 24, 32, 2  # group gn; data sdn-edn; metal stn-endn
xlsxf = 'Cleaned_data.xlsx'

current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_excel(os.path.join(current_dir, xlsxf), sheet_name="model", engine='openpyxl')
print(df.columns.tolist())

metal = ['W', 'Sn', 'Mo', 'Fe', 'Cu', 'Au', 'Pb', 'Zn']
for i in range(stn, endn):
    # W 24, Sn 25, Mo 26, Fe 27, Cu 28, Au 29, Pb 30, Zn 31
    if i in [24, 26, 27, 28, 30]:
        Ml = 'LightGBM'
    elif i in [25, 29, 31]:
        Ml = 'XGBoost'

    if i in [27, 29, 31]:
        log = 'raw'
    elif i in [24, 25, 26, 28, 30]:
        log = 'log10'

    if i in [28]:
        sp = 'weighting'
    elif i in [24, 25, 26, 27, 29, 30, 31]:
        sp = 'smote'

    if log == 'log10':
        x, xkeys = np.log10(df.iloc[:, sdn:edn]), df.columns[sdn:edn].tolist()
    elif log == 'raw':
        x, xkeys = df.iloc[:, sdn:edn], df.columns[sdn:edn].tolist()

    ctn = i - stn
    metal_cur = metal[i - edn]
    print(metal_cur, Ml, log, sp, i, ctn)

    y = df.iloc[:, i]
    unique, counts = np.unique(y, return_counts=True)
    print(f"Data set:")
    for value, count in zip(unique, counts):
        print(f"{value}: {count} ")

    # <editor-fold desc="Data split">
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    # </editor-fold>

    # <editor-fold desc="Class imbalance">
    if sp == 'smote':
        x_train0, y_train0, x_test0, y_test0 = x_train, y_train, x_test, y_test
        x_train, y_train = SMOTE(random_state=42).fit_resample(x_train0, y_train0)
        x_test, y_test = SMOTE(random_state=42).fit_resample(x_test0, y_test0)
    elif sp == 'weighting':
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
        class_weight_map = dict(zip(unique_classes, weights))
        sample_weights = np.array([class_weight_map[label] for label in y_train])

    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Train set:")
    for value, count in zip(unique, counts):
        print(f"{value}: {count} ")
    unique, counts = np.unique(y_test, return_counts=True)
    print(f"Test set:")
    for value, count in zip(unique, counts):
        print(f"{value}: {count} ")
    # </editor-fold>

    # <editor-fold desc="Models and BayesSearchCV">
    if Ml == 'XGBoost':
        clf = xgb.XGBClassifier(
            random_state=42,
            objective='binary:logistic',
            importance_type='gain',
            eval_metric='logloss', )
        clf_cv = copy.deepcopy(clf)
        param_space = {'colsample_bytree': Real(0.05, 1.0),
                       'gamma': Real(0.05, 1),
                       'learning_rate': Real(0.05, 1.0),
                       'reg_alpha': Real(0.05, 1),
                       'reg_lambda': Real(0.05, 1),
                       'subsample': Real(0.05, 1),
                       'max_depth': Integer(1, 10),
                       'min_child_weight': Integer(1, 10),
                       'n_estimators': Integer(500, 1000), }
        column_names = ['colsample_bytree',
                        'gamma',
                        "learning_rate",
                        'reg_alpha',
                        'reg_lambda',
                        'subsample',
                        "max_depth",
                        "min_child_weight",
                        'n_estimators',
                        "score", ]
    elif Ml == 'LightGBM':
        feature_names = df.columns[sdn:edn].tolist()
        train_data = lgb.Dataset(x_train, label=y_train, feature_name=feature_names)
        test_data = lgb.Dataset(x_test, label=y_test, feature_name=feature_names)
        params = {
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'num_class': 2,
            'verbose': -1,
            'random_state': 42,
            'force_col_wise': True,
        }

        clf = lgb.LGBMClassifier(**params)

        clf_cv = copy.deepcopy(clf)
        param_space = {
            'learning_rate': Real(0.01, 0.9),
            'subsample': Real(0.01, 0.9),
            'colsample_bytree': Real(0.01, 0.9),
            'reg_alpha': Real(0.01, 0.9),
            'reg_lambda': Real(0.01, 0.9),
            'max_depth': Integer(2, 10),
            'num_leaves': Integer(4, 1024),
            'n_estimators': Integer(500, 1000),
        }
        column_names = [
            'learning_rate',
            'subsample',
            'colsample_bytree',
            'reg_alpha',
            'reg_lambda',
            'max_depth',
            'num_leaves',
            'n_estimators',
            "score", ]
    scoring = make_scorer(f1_score, average='binary')
    opt = BayesSearchCV(clf, param_space, random_state=42, n_iter=5, cv=3,
                        scoring=scoring, n_jobs=-1, verbose=0, )

    x_combined = pd.DataFrame(np.row_stack((x_train, x_test)), columns=feature_names)
    y_combined = np.concatenate((y_train, y_test))
    opt.fit(x_combined, y_combined)

    search_res = np.column_stack((np.array(opt.optimizer_results_[0][f'x_iters']),
                                  -1 * opt.optimizer_results_[0][f'func_vals']))
    search_res_df = pd.DataFrame(search_res, columns=column_names)
    clf.set_params(**opt.best_params_)
    if sp == 'weighting':
        clf.fit(x_train, y_train, sample_weight=sample_weights)
    else:
        clf.fit(x_train, y_train)
    pkl_path = os.path.join(current_dir, Ml) + '_' + log + '_' + sp + '_' + metal[ctn] + '.bin'
    # with open(pkl_path, 'wb') as f:
    #     pickle.dump(clf, f)
    # </editor-fold>

    # <editor-fold desc="Test set result">
    y_pred = clf.predict(x_test)
    cmatrix = confusion_matrix(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred)
    pre_score = precision_score(y_test, y_pred)
    rec_score = recall_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)
    cmatrix = confusion_matrix(y_test, y_pred)
    y_proba = clf.predict_proba(x_test)
    auc1 = roc_auc_score(y_test, y_proba[:, 1])

    print(cmatrix)
    print(classification_report(y_test, y_pred))
    print(f"Accuracy\t{acc_score:.4f}", )
    print(f"Precision\t{pre_score:.4f}", )
    print(f"Recall\t{rec_score:.4f}", )
    print(f"F1-score\t{f1score:.4f}", )
    print(f"AUC\t{auc1:.4f}", )
    # </editor-fold>

    # <editor-fold desc="Kfold Cross-validation">
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    f1_scores = []
    auc0_scores = []
    auc1_scores = []
    for train_index, val_index in skf.split(x, y):
        x_train_fold, x_val_fold = x.iloc[train_index], x.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        if sp == 'smote':
            x_train_fold_res, y_train_fold_res = SMOTE(random_state=42).fit_resample(x_train_fold, y_train_fold)
            x_val_fold_res, y_val_fold_res = SMOTE(random_state=42).fit_resample(x_val_fold, y_val_fold)
        elif sp == 'weighting':
            unique_classes, class_counts = np.unique(y_train_fold, return_counts=True)
            weights = compute_class_weight('balanced', classes=unique_classes, y=y_train_fold)
            sample_weights = np.ones(len(x_train_fold))
            for class_label, weight in enumerate(weights):
                sample_weights[y_train_fold == class_label] = weight
            x_val_fold_res, y_val_fold_res = x_val_fold, y_val_fold
        clf_cv.set_params(**opt.best_params_)
        if sp == 'weighting':
            clf_cv.fit(x_train_fold, y_train_fold, sample_weight=sample_weights)
        else:
            clf_cv.fit(x_train_fold_res, y_train_fold_res)
        y_val_pred = clf.predict(x_val_fold_res)
        fold_f1 = f1_score(y_val_fold_res, y_val_pred, average='binary')
        y_val_pred = clf_cv.predict(x_val_fold_res)
        fold_f1 = f1_score(y_val_fold_res, y_val_pred, average='binary')
        f1_scores.append(fold_f1)
    y_val_proba = clf_cv.predict_proba(x_val_fold_res)
    fold_auc0 = roc_auc_score(1 - y_val_fold_res, y_val_proba[:, 0])
    auc0_scores.append(fold_auc0)
    fold_auc1 = roc_auc_score(y_val_fold_res, y_val_proba[:, 1])
    auc1_scores.append(fold_auc1)
    print(f"F1-Score_cv\t{np.mean(f1_scores):.4f}\n")
    # </editor-fold>
    del clf, clf_cv
