#!/usr/bin/env python
### author: Jingmei Yang: jmyang@bu.edu
import pandas as pd
import os
from model import ModelInitializer, DataImputer, DataScaler
import numpy as np
from log_info import setup_logger
from extraction_funs import read_file2list, save_columns
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold
from sklearn.feature_selection import RFECV
from funs import save_2pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define Config
random_state = 42

param_dict = {'logistic_regression': {
    'solver': ['liblinear'],
    'penalty': ['l1', 'l2'],
    'C': np.logspace(-4, 4, 20),
    'max_iter': [100, 200, 300, 400, 500, 600, 700, 800, 900,1000]
},
'xgboost':{
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 6, 9, 12],
    'gamma': [0, 0.1, 0.2, 0.3]
},
'decision_tree':{
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'max_features': [ 'auto', 'sqrt', 'log2']},
'svm':{
    'C': [0.1, 1, 10, 100, 1000],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10, 100],
    'shrinking': [True, False],
    'probability': [True, False]
}
}



# ['logistic_regression','decision_tree', 'xgboost']
model_type = 'logistic_regression'
param_grid = param_dict[model_type]
use_rfe = True

output_path = os.path.join('./', model_type)
if not os.path.exists(output_path):
    os.makedirs(output_path)
n_folds = 5
logger = setup_logger("feature selection", f"{model_type}_feature_selection.log")
df_final= pd.read_csv("preprocessed_per_np.csv")
target_col = 'target'
feature_cols=read_file2list("keep_np_tests.txt")


# Split data
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
splits = list(skf.split(df_final, df_final[target_col]))
save_2pickle('cross_splits.pkl', splits)

# Impute and normalize data for each fold
for i, (train_index, test_index) in enumerate(splits):
    train_raw = df_final.loc[train_index].copy()
    test_raw = df_final.loc[test_index].copy()
    inference_raw = pd.read_csv("inference_test_preprocessed.csv")

    subfolder = os.path.join(output_path, f"Fold_{i}")
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    # Fill missing value
    data_imputer= DataImputer(subfolder)
    train_data_imputed = data_imputer.process_data(train_raw, feature_cols, fit_imputer=True, normalizer_file='imputation.json')
    test_data_imputed  = data_imputer.process_data(test_raw, feature_cols, fit_imputer=False, normalizer_file='imputation.json')
    inference_imputed = data_imputer.process_data(inference_raw, feature_cols, fit_imputer=False, normalizer_file='imputation.json')
    # Normalize the numerical variables
    data_scaler = DataScaler(subfolder)
    train_data = data_scaler.process_data(train_data_imputed, feature_cols, fit_scaler=True, scaler_file='scaler.pkl')
    test_data = data_scaler.process_data(test_data_imputed, feature_cols, fit_scaler=False, scaler_file='scaler.pkl')
    inference_data = data_scaler.process_data(inference_imputed, feature_cols, fit_scaler=False, scaler_file='scaler.pkl')

    # Save imputed and normalized dataset for future use
    train_data.to_csv(os.path.join(subfolder, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(subfolder, 'test.csv'), index=False)
    inference_data.to_csv(os.path.join(subfolder, 'inference.csv'), index=False)


# Read the entire data
all_data = pd.concat([pd.read_csv(os.path.join(output_path, f"Fold_{i}", 'test.csv')) for i in range(5)])

# Define the model
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=random_state)

#  Perform recursive feature elimination
if use_rfe:
    model_initializer = ModelInitializer(model_type, hyperparameters=None, random_state=random_state)
    model = model_initializer.initialize_model()

    selector = RFECV(estimator=model, step=1, cv=cv, scoring='roc_auc')
    selector = selector.fit(all_data[feature_cols], all_data[target_col])
    selected_features = [f for f, s in zip(feature_cols, selector.support_) if s]
    save_columns(selected_features, file_name=f"rfe_selected_features_per_np.txt")

else:
    selected_features = read_file2list("rfe_selected_features_per_np.txt")


# Perform hyperparameter tuning
model_initializer = ModelInitializer(model_type, hyperparameters=None, random_state=random_state)
model = model_initializer.initialize_model()
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='roc_auc')
grid_search.fit(all_data[selected_features], all_data[target_col])

best_params = grid_search.best_params_
best_score = grid_search.best_score_
save_2pickle(f'{model_type}_best_params_per_np.pkl', best_params)
logger.info(f"best_params:\n {best_params}")
logger.info(f"best_score:\n {best_score}")
logger.info(f"selected_features:\n {selected_features}")


