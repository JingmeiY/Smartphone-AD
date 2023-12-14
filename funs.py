#!/usr/bin/env python
### author: Jingmei Yang: jmyang@bu.edu
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('mode.chained_assignment', None)
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import pickle
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve, fbeta_score, classification_report, roc_auc_score)



def create_density_plot(data, file_name):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, color="blue", bins=30)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.savefig(file_name, dpi=300)
    plt.show()


def save_2pickle(pkl_file, content):
    with open(pkl_file, 'wb') as f:
        pickle.dump(content, f)

def load_pickle(pkl_file):
    with open(pkl_file, 'rb') as f:
        content = pickle.load(f)
    return content


def count_percentage(df, col):
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in the DataFrame.")

    label_counts = df[col].value_counts()

    label_percentage = label_counts / label_counts.sum() * 100

    label_summary = {
        'Level': label_counts.index,
        'Count': label_counts.values,
        'Percentage': label_percentage.values.round(2)}

    return pd.DataFrame(label_summary)


def missing_values_summary(df, columns, threshold):
    column_names = []
    percentages = []
    less_threshold_columns = []
    total_rows = len(df)

    for col in columns:
        missing = df[col].isnull().sum()
        percentage = (missing / total_rows) * 100

        if percentage < threshold:
            less_threshold_columns.append(col)
        column_names.append(col)
        percentages.append(percentage)


    results = pd.DataFrame({
        'Column': column_names,
        'Percentage': percentages
    })

    results_sorted = results.sort_values(by='Percentage', ascending=True)

    return results_sorted, less_threshold_columns

def read_sas(sas_file):
    data = pd.read_sas(sas_file)
    data['id'] = data['idtype'].astype(int).astype(str) + "_" + data['id'].astype(int).astype(str)
    return data



class DatasetInspector:

    def __init__(self, df, logger = None):
        self.df = df
        self.logger = logger or logging.getLogger('data_preprocessing')


    def inspect_column(self, column_name):

        self.logger.info(f"\n\nColumn Name: {column_name}")
        self.logger.info(f"Data Type: {self.df[column_name].dtype}")
        self.logger.info(f"Count: {self.df[column_name].shape[0]}")
        self.logger.info(f"Number of Missing Values: {self.df[column_name].isnull().sum()}")

        if pd.api.types.is_numeric_dtype(self.df[column_name]):
            self.logger.info(f"Mean: {self.df[column_name].mean():.2f}")
            self.logger.info(f"Standard Deviation: {self.df[column_name].std():.2f}")
            self.logger.info(f"Median: {self.df[column_name].median():.2f}")
            self.logger.info(f"Minimum: {self.df[column_name].min():.2f}")
            self.logger.info(f"Maximum: {self.df[column_name].max():.2f}")

        elif pd.api.types.is_categorical_dtype(self.df[column_name]) or self.df[column_name].dtype == 'object':
            counts, percentages = self._get_categorical_distribution(column_name)
            self.logger.info(pd.DataFrame({'Counts': counts, 'Percentage': percentages}))


    def _get_categorical_distribution(self, column_name):
        counts = self.df[column_name].value_counts(dropna=False)  # Handle NaN values too
        percentages = counts / len(self.df) * 100
        return counts, percentages.round(2)

class ThresholdOptimizer:

    def __init__(self, y_true, y_prob,
                 metric_function_name="f1_score", average='binary',
                drop_intermediate=False):
        self.y_true = np.array(y_true)
        self.y_prob = np.array(y_prob)
        self.metric_function_name = metric_function_name
        self.average = average
        self.drop_intermediate = drop_intermediate

    def optimize_threshold(self):
        fpr, tpr, roc_thresholds = roc_curve(self.y_true, self.y_prob, drop_intermediate=self.drop_intermediate)
        thresholds = roc_thresholds[1:]

        if self.metric_function_name == "G-mean":
            scores = np.sqrt(tpr * (1-fpr))
        elif self.metric_function_name == "fbeta_score":
            scores = [fbeta_score(self.y_true, (self.y_prob > thr).astype(int).tolist(), beta=2.0) for thr in thresholds]
        else:
            metric_function = getattr(metrics, self.metric_function_name)
            scores = [metric_function(self.y_true, (self.y_prob > thr).astype(int).tolist(), average=self.average) for thr in thresholds]

        best_score_index = np.argmax(scores)
        optimized_threshold = thresholds[best_score_index]
        return optimized_threshold

    def get_prediction(self, optimize_threshold_flag=True):
            if optimize_threshold_flag:
                threshold = self.optimize_threshold()
            else:
                threshold = 0.5

            print(f"Threshold: {threshold}")
            return threshold, np.array(1* (self.y_prob > threshold))


class ModelEvaluator:
    def __init__(self, output_path,
                 average='binary',
                 calculate_metrics_flag=True,
                 plot_confusion_matrix_flag=True,
                 plot_roc_curve_flag=True,
                 save_classification_report_flag=True,
                 plot_config = None):
        self.average = average
        os.makedirs(output_path, exist_ok=True)
        self.output_path = output_path
        self.calculate_metrics_flag = calculate_metrics_flag
        self.plot_confusion_matrix_flag = plot_confusion_matrix_flag
        self.plot_roc_curve_flag = plot_roc_curve_flag
        self.save_classification_report_flag = save_classification_report_flag
        self.config = plot_config or  {"fig_size": (10, 8), "dpi": 300, "format": ".jpg", "title_fontsize": 16, "label_fontsize": 14, "tick_fontsize": 12}

    def calculate_metrics(self, y_true, y_pred, y_prob=None,
                          output_path=None, file_name="metrics.csv"):
        metrics = {}
        if y_prob is not None:
            metrics['roc_auc'] =roc_auc_score(y_true, y_prob)
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            metrics['pr_auc'] = auc(recall, precision)
        else:
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None

        metrics['recall'] = recall_score(y_true, y_pred, average="binary", zero_division=0)
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        total_negative = cm[0, 0] + cm[0, 1]
        metrics['specificity'] = cm[0, 0] / total_negative if total_negative != 0 else 0
        metrics['precision'] = precision_score(y_true, y_pred, average="binary", zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average="binary", zero_division=0)

        metrics = {
            key: (round(value * 100, 2) if value is not None else None)
            for key, value in metrics.items()
        }
        print(metrics)
        df = pd.DataFrame([metrics])
        df.to_csv(os.path.join((output_path or self.output_path), f'{file_name}'), index=False)

    def save_classification_report(self, y_true, y_pred,
                                   output_path=None, file_name="classification_report.txt", title="Classification Report"):
        report = classification_report(y_true, y_pred, zero_division=0)
        with open(os.path.join((output_path or self.output_path), file_name), "w") as f:
            f.write(f"\n{title}\n{report}")


    def plot_confusion_matrix(self, y_true = None, y_pred = None,
                              cmatrix= None, labels_names = ["0", "1"], normalize=True,
                              output_path=None, file_name="confusion_matrix", title="Confusion Matrix"):

        cm = np.array(cmatrix) if cmatrix is not None else confusion_matrix(y_true, y_pred)


        plt.figure(figsize=self.config["fig_size"])

        if normalize:
            cm_sum = cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = cm.astype('float') / cm_sum
            annot = np.empty_like(cm, dtype='<U20')
            annot[0, 0] = f'TN: {cm[0, 0]} ({cm_normalized[0, 0]:.2%})'
            annot[0, 1] = f'FP: {cm[0, 1]} ({cm_normalized[0, 1]:.2%})'
            annot[1, 0] = f'FN: {cm[1, 0]} ({cm_normalized[1, 0]:.2%})'
            annot[1, 1] = f'TP: {cm[1, 1]} ({cm_normalized[1, 1]:.2%})'
        else:
            annot = cm.astype('<U20')
        print(f"confusion matrix{annot}")
        sns.heatmap(cm, annot=annot, fmt='', xticklabels=labels_names, yticklabels=labels_names, cmap="Blues")

        plt.title(f"{title}", fontsize=self.config["title_fontsize"])
        plt.xlabel('Predicted label', fontsize=self.config["label_fontsize"])
        plt.ylabel('True label', fontsize=self.config["label_fontsize"])
        plt.xticks(fontsize=self.config["tick_fontsize"])
        plt.yticks(fontsize=self.config["tick_fontsize"])
        plt.tight_layout()
        plt.savefig(os.path.join((output_path or self.output_path), f'{file_name}{self.config["format"]}'), dpi=self.config["dpi"])
        plt.close()




    def plot_roc_curve(self, y_true, y_prob,
                       output_path=None, file_name="roc_curve", title="Receiver Operating Characteristic"):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=self.config["fig_size"])
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label='Random AUC = 0.5')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=self.config["label_fontsize"])
        plt.ylabel("True Positive Rate", fontsize=self.config["label_fontsize"])
        plt.title(f"{title}", fontsize=self.config["title_fontsize"])
        plt.legend(loc="lower right", borderpad=0.5, labelspacing=0.5, handlelength=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join((output_path or self.output_path), f'{file_name}{self.config["format"]}'), dpi=self.config["dpi"])
        plt.close()

    def run_all(self, y_true, y_pred, y_prob=None, labels_names = ["0", "1"]):
        if self.calculate_metrics_flag:
            self.calculate_metrics(y_true = y_true, y_pred = y_pred, y_prob = y_prob)

        if self.save_classification_report_flag:
            self.save_classification_report(y_true = y_true, y_pred = y_pred)

        if self.plot_confusion_matrix_flag:
            self.plot_confusion_matrix(y_true = y_true, y_pred = y_pred, labels_names = labels_names)

        if self.plot_roc_curve_flag:
            self.plot_roc_curve(y_true = y_true, y_prob = y_prob)
