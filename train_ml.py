#!/usr/bin/env python
### author: Jingmei Yang: jmyang@bu.edu

import pandas as pd
import os
from model import  ModelTrainer, ModelPredictor
from funs import ThresholdOptimizer, ModelEvaluator, load_pickle, save_2pickle
import numpy as np
from log_info import setup_logger
from extraction_funs import read_file2list
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

random_state = 42
# ['logistic_regression','decision_tree','xgboost']
model_type = 'logistic_regression'
optimize_threshold_flag = True
output_path = os.path.join('./', model_type)
if not os.path.exists(output_path):
    os.makedirs(output_path)

logger = setup_logger("fitting", f"{model_type}_fitting.log")
n_folds = 5
target_col = 'target'
selected_features = read_file2list("rfe_selected_features_per_np.txt")
best_params = load_pickle(f'{model_type}_best_params_per_np.pkl')

threshold_dict = {}
for i in range(n_folds):
    subfolder = os.path.join(output_path, f"Fold_{i}")

    train_data = pd.read_csv(os.path.join(subfolder, 'train.csv'))
    test_data = pd.read_csv(os.path.join(subfolder, 'test.csv'))

    # Train model
    model_trainer = ModelTrainer(model_type = model_type, hyperparameters = best_params, output_path = subfolder,
                             random_state = random_state)

    model, model_path = model_trainer.train_model(train_data = train_data, feature_cols = selected_features,
                                                  target_col = target_col, save_model = True)


    # Predict for test data
    model_predictor = ModelPredictor(model_path = model_path,  feature_cols = selected_features)
    y_true = np.array(test_data[target_col].values.astype(int))

    if hasattr(model_predictor.model, 'predict_proba'):
        y_prob = model_predictor.predict_proba(test_data)
        threshold_optimizer = ThresholdOptimizer(y_true = y_true, y_prob = y_prob,
                                                 metric_function_name = "f1_score",
                                                 average = 'binary', drop_intermediate = False)

        threshold, y_pred = threshold_optimizer.get_prediction(optimize_threshold_flag = optimize_threshold_flag)
        y_pred = np.array(y_pred.astype(int))
        threshold_dict[f"Fold_{i}"] = threshold

    else:
        y_pred = model_predictor.predict(test_data)
        y_prob = None


    #  Evaluate model
    test_evaluator = ModelEvaluator(
        output_path = subfolder, average = 'binary',
        calculate_metrics_flag = True, plot_confusion_matrix_flag = True, plot_roc_curve_flag = True, save_classification_report_flag = True,
        plot_config = None)

    test_evaluator.run_all(y_true = y_true, y_pred = y_pred, y_prob = y_prob)

save_2pickle(f'{model_type}_threshold.pkl', threshold_dict)

