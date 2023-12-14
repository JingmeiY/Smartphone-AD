#!/usr/bin/env python
### author: Jingmei Yang: jmyang@bu.edu
import pickle
import joblib
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

class ModelInitializer:
    def __init__(self, model_type, hyperparameters=None, random_state=123):
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.default_hyperparams = self._set_default_hyperparameters()
        self.user_hyperparams = hyperparameters if hyperparameters else {}

    def _set_default_hyperparameters(self):
        default_hyperparams = {
            'logistic_regression': {'random_state': self.random_state},
            'svm': {'random_state': self.random_state},
            'decision_tree': {'random_state': self.random_state},
            'random_forest': {'random_state': self.random_state},
            'xgboost': {'random_state': self.random_state}
        }
        return default_hyperparams

    def initialize_model(self):
        if self.model_type not in self.default_hyperparams:
            raise ValueError(f"Model type {self.model_type} not recognized.")
        hyperparams = {**self.default_hyperparams[self.model_type], **self.user_hyperparams}

        if self.model_type == 'logistic_regression':
            return LogisticRegression(**hyperparams)
        elif self.model_type == 'svm':
            return SVC(**hyperparams)
        elif self.model_type == 'decision_tree':
            return DecisionTreeClassifier(**hyperparams)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(**hyperparams)
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(**hyperparams)



class DataImputer:
    def __init__(self, output_path = './'):
        self.output_path = output_path
        self.imputation_values = None

    def process_data(self, data, columns, fit_imputer = True, normalizer_file = 'imputation.json'):
        if fit_imputer:
            data, self.imputation_values = self._fill_missing_value(data, columns)
            self._save_imputation_values(normalizer_file)
        else:
            self._load_imputation_values(normalizer_file)
            data = self._fill_with_predefined_values(data, columns, self.imputation_values)
        return data

    def _fill_missing_value(self, df, columns):
        imputation_values = {}
        for column in columns:
            if df[column].dtype == 'float64' or df[column].dtype == 'int64':
                mean_value = df[column].mean()
                df[column].fillna(mean_value, inplace = True)
                imputation_values[column] = mean_value
            else:
                mode_value = df[column].mode()[0]
                df[column].fillna(mode_value, inplace = True)
                imputation_values[column] = mode_value
        return df, imputation_values

    def _fill_with_predefined_values(self, df, columns, imputation_values):
        for column in columns:
            if column in imputation_values:
                df[column].fillna(imputation_values[column], inplace = True)
        return df

    def _save_imputation_values(self, file_name):
        if not self.output_path:
            raise ValueError("Output path not set for saving the imputation values.")
        file_path = os.path.join(self.output_path, file_name)
        with open(file_path, 'w') as file:
            json.dump(self.imputation_values, file)

    def _load_imputation_values(self, file_name):
        file_path = os.path.join(self.output_path, file_name)
        with open(file_path, 'r') as file:
            self.imputation_values = json.load(file)



class DataScaler:
    def __init__(self, output_path=''):
        self.output_path = output_path
        self.scaler = MinMaxScaler()

    def process_data(self, data, numerical_features, fit_scaler = True, scaler_file = 'scaler.pkl'):
        if fit_scaler:
            self.scaler.fit(data[numerical_features])
            data[numerical_features] = self.scaler.transform(data[numerical_features])
            self._save_scaler(scaler_file)
        else:
            self._load_scaler(scaler_file)
            data[numerical_features] = self.scaler.transform(data[numerical_features])
        return data

    def _save_scaler(self, file_name):
        if not self.output_path:
            raise ValueError("Output path not set for saving the scaler.")
        file_path = os.path.join(self.output_path, file_name)
        with open(file_path, 'wb') as file:
            pickle.dump(self.scaler, file)

    def _load_scaler(self, file_name):
        file_path = os.path.join(self.output_path, file_name)
        with open(file_path, 'rb') as file:
            self.scaler = pickle.load(file)


class ModelTrainer:
    def __init__(self, model_type, hyperparameters = None, output_path = './', random_state=None):
        self.model_initializer = ModelInitializer(model_type, hyperparameters, random_state)
        self.model = self.model_initializer.initialize_model()
        self.output_path = output_path
        self.model_type = model_type

    def train_model(self, train_data, feature_cols, target_col, save_model = False):
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        self.model.fit(X_train, y_train)
        model_path = os.path.join(self.output_path, f"{self.model_type}_model.pkl")
        if save_model:
            joblib.dump(self.model, model_path)
        return self.model, model_path




class ModelPredictor:
    def __init__(self, model_path,  feature_cols):
        self.model_path = model_path
        self.feature_cols = feature_cols
        self.model = self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

    def predict(self, input_data):
        predictions = self.model.predict(input_data[self.feature_cols])
        return predictions

    def predict_proba(self, input_data):
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(input_data[self.feature_cols])[:, 1]
            return probabilities
        else:
            raise AttributeError("This model does not support probability predictions")









