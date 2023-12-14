#!/usr/bin/env python
### author: Jingmei Yang: jmyang@bu.edu

import pandas as pd
import os
from model import  ModelPredictor
from funs import  load_pickle, create_density_plot
import numpy as np
from log_info import setup_logger
from extraction_funs import read_file2list
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

random_state = 42
# ['logistic_regression','decision_tree','xgboost']
model_type = 'logistic_regression'
output_path = os.path.join('./', model_type)
if not os.path.exists(output_path):
    os.makedirs(output_path)


logger = setup_logger("prediction", f"{model_type}_prediction.log")
n_folds = 5
selected_features=read_file2list("rfe_selected_features_per_np.txt")
feature_cols = read_file2list("keep_np_tests.txt")
threshold_dict =  load_pickle(f'{model_type}_threshold.pkl')
inference_raw = pd.read_csv("inference_test_preprocessed.csv")

for i in range(n_folds):
    # Load trained model
    subfolder = os.path.join(output_path, f"Fold_{i}")
    model_predictor = ModelPredictor(model_path = os.path.join(subfolder, f"{model_type}_model.pkl"), feature_cols = selected_features)
    test_data = pd.read_csv(os.path.join(subfolder, 'inference.csv'))


    # Predict for test data
    y_prob = model_predictor.predict_proba(test_data)
    y_pred =  model_predictor.predict(test_data)
    print( threshold_dict[f"Fold_{i}"])
    y_pred_threshold = np.array(1* (y_prob > threshold_dict[f"Fold_{i}"]))
    inference_raw[f"prob_{i}"] = y_prob
    inference_raw[f"pred_{i}"] = y_pred
    inference_raw[f"pred_threshold_{i}"] = y_pred_threshold
    create_density_plot(y_prob, os.path.join(subfolder, "density.png"))

inference_raw.to_csv("inference_test_predicted.csv", index = False)
