#!/usr/bin/env python
### author: Jingmei Yang: jmyang@bu.edu

import pandas as pd
import re
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('mode.chained_assignment', None)
import numpy as np
from funs import read_sas
from log_info import setup_logger
from extraction_funs import filter_columns_with_string_and_suffix, extend_columns_with_suffix,  read_file2list

def melt_dataframe(df, num_np):
    melt_data = []
    for _, row in df.iterrows():
        for np in range(1, num_np + 1):
            new_row = {col: row[col] for col in df.columns if "_NP" not in col}
            new_row.update({col.replace(f"_NP{np}", ""): row[col] for col in df.columns if col.endswith(f"_NP{np}")})
            melt_data.append(new_row)
    melt_df = pd.DataFrame(melt_data)
    return melt_df


def assign_status_closest_np(row, status_date_cols):
    if row[status_date_cols].isna().all():
      return "non_review", np.nan

    else:
      days_difference = {col: abs(row[col] - row['examdate']).days for col in status_date_cols if pd.notna(row[col])}
      status = min(days_difference, key=days_difference.get)
      return status.split("_")[0], days_difference[status]

# Setup Config
Gen = {"Gen1": "./Curated_datasets/Curated_datasets/Gen1/curated_bap_0_0919.sas7bdat",
"Gen2": "./Curated_datasets/Curated_datasets/Gen2_Omni1/curated_bap_17_0712.sas7bdat",
"Gen3": "./Curated_datasets/Curated_datasets/Gen3_NOS_Omni2/curated_bap_2372_0712.sas7bdat"}
Gen_selected = 2
sas_file = (Gen[f"Gen{Gen_selected}"])
df = read_sas(sas_file)

# Setup logger
logger = setup_logger("data_extraction", f"extract_Gen{Gen_selected}.log")

# Read in the selected np tests
selected_tests = read_file2list(file = "NP_variables.txt")

# Preprocess cognitive status related variables
status_date_cols=["normal_date", "impairment_date", "mild_date", "moderate_date", "severe_date"]
for col in status_date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Preprocess NP exam date related variables
examdate = filter_columns_with_string_and_suffix(df.columns, ["examdate"], suffix_pattern='_NP')
for col in examdate:
    df[col] = pd.to_datetime(df[col], errors='coerce')
num_np = max([int(re.search(r'NP(\d+)', date).group(1)) for date in examdate])
logger.info(f"######Total number of NP measurements- {num_np}")


# Select the variables with the _NP suffix, status related, age, and id
NP1_N = extend_columns_with_suffix(selected_tests+["age"], num_np)
selected_cols = examdate + ["id"] + status_date_cols + NP1_N
filtered_df = df[selected_cols].copy()

# Extract NP exams for each patient
melted_df = melt_dataframe(filtered_df, num_np)
melted_df_cleaned = melted_df.dropna(subset=["examdate"])

# Assign a  flag to each NP exam
melted_df_cleaned['flag'], melted_df_cleaned['days_flag_np'] = zip(*melted_df_cleaned.apply(lambda row: assign_status_closest_np(row, status_date_cols), axis=1))
melted_df_cleaned.to_csv(f"melted_data_Gen{Gen_selected}.csv", index = False)






