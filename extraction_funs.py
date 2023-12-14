
#!/usr/bin/env python
### author: Jingmei Yang: jmyang@bu.edu

import pandas as pd
import re
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('mode.chained_assignment', None)
from funs import  missing_values_summary

def read_file2list(file):
    with open(file, "r") as file:
        list = [line.strip() for line in file]
    return list

def save_columns(save_list, file_name = f"keep_columns_Gen2_less10.txt"):
    with open(file_name, "w") as file:
        for item in save_list:
            file.write(item + "\n")

def inspect_features(df, columns, suffix_pattern='_NP', threshold = 90):
    features = filter_columns_with_string_and_suffix(df.columns, columns, suffix_pattern=suffix_pattern)
    df_temp = df[features].copy()
    results_sorted, less_threshold_columns = missing_values_summary(df_temp, features, threshold = threshold)
    print(less_threshold_columns)




def generate_suffix_pattern(np_num, max_np):
    if np_num > max_np:
        raise ValueError("np_num should be less than or equal to max_np")

    if np_num < 10:
        return rf'_NP[1-{np_num}]$'
    else:
        single_digit_pattern = 'NP[1-9]'
        double_digit_pattern = '|'.join([f'NP{d}' for d in range(10, np_num + 1)])

        return rf'_({single_digit_pattern}|{double_digit_pattern})$'




def extract_np_for_missing(df, default_np=1):
    result = []
    for row in df.itertuples(index=False):
        data_dict = {}
        for col in df.columns:
            if f'_NP{default_np}' in col:
                new_col_name = col.replace(f'_NP{default_np}', '')
                data_dict[new_col_name] = getattr(row, col)
            elif "_NP" not in col:
                data_dict[col] = getattr(row, col)
        result.append(data_dict)
    return pd.DataFrame(result)


def get_severity(row, severity_order = ['severe', 'moderate', 'mild', 'impairment', 'normal'] ):

    stages = {
        'normal': row['normal_date'],
        'impairment': row['impairment_date'],
        'mild': row['mild_date'],
        'moderate': row['moderate_date'],
        'severe': row['severe_date']
    }

    valid_stages = {stage: date for stage, date in stages.items() if pd.notna(date)}
    for stage in severity_order:
        if stage in valid_stages:
            return stage, valid_stages[stage]
    return "probable normal", None


def find_closest_exam_np(row, severity_date_col='severity_date', num_np=4):

    severity_date = pd.to_datetime(row[severity_date_col], errors='coerce')
    if pd.isna(severity_date):
        return "Missing", float('inf')

    closest_np = "Missing"
    min_diff = float('inf')

    for i in range(1, num_np + 1):
        exam_date_col = f'examdate_NP{i}'
        exam_date = pd.to_datetime(row[exam_date_col], errors='coerce')

        if pd.notna(exam_date):
            diff = abs((severity_date - exam_date).days)
            if diff < min_diff:
                min_diff = diff
                closest_np = f'NP{i}'

    return closest_np, min_diff


def filter_columns_with_string_and_suffix(columns, strings, suffix_pattern):

    filtered_columns = [
        col for col in columns
        if any(string in col for string in strings) and re.search(suffix_pattern, col)
    ]
    return filtered_columns


def extend_columns_with_suffix(columns, np_num):
    extended_columns = []
    for col in columns:
        for np in range(1, np_num + 1):
            extended_columns.append(f"{col}_NP{np}")
    return extended_columns


def extract_and_reshape(df, closest_exam_np_col = "closest_exam_np"):

    result = []
    for row in df.itertuples(index=False):
        closest_exam_np = getattr(row, closest_exam_np_col)
        if closest_exam_np == "Missing":
            continue
        data_dict = {}
        for col in df.columns:
            if closest_exam_np in col:
                new_col_name = col.split('_' + closest_exam_np)[0]
                data_dict[new_col_name] = getattr(row, col)

            elif not "_NP" in col:
                data_dict[col] = getattr(row, col)

        result.append(data_dict)

    return pd.DataFrame(result)




