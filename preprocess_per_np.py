#!/usr/bin/env python
### author: Jingmei Yang: jmyang@bu.edu

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('mode.chained_assignment', None)
from log_info import setup_logger
from extraction_funs import  read_file2list, save_columns
from funs import missing_values_summary, count_percentage
random_state = 42

def map_label(df, map_dict, status_col, target_col):
    df_filtered= df[df[status_col].isin(map_dict.keys())].copy()
    df_filtered[target_col] = df_filtered[status_col].map(map_dict)
    print(df_filtered[target_col].value_counts())
    return df_filtered


# Setup
logger = setup_logger("data_preprocess", "process.log")
threshold = 365 * 1
use_non_review = True

# Read the Data from Gen 2 and Gen 3, and combine them into one dataframe
selected_tests = read_file2list(file = "NP_variables.txt") + ["age"]

df_G2 = pd.read_csv(f"melted_data_Gen2.csv")
df_G3= pd.read_csv(f"melted_data_Gen3.csv")
df_combined = pd.concat([df_G2, df_G3])
df_combined['examdate'] = pd.to_datetime(df_combined['examdate'], errors='coerce')
df_combined['combined_key'] = df_combined['id'].astype(str) + '_' + df_combined['examdate'].astype(str)


# Read smartphone data and extract the examdate_np and id
smartphone = pd.read_csv("np_smartphone_examdates.csv")[['ptid', 'examdate_np']]

# Remove duplicates from Smartphone patients based on the id and examdate_np
smartphone_unique = smartphone.drop_duplicates(subset=['ptid', 'examdate_np'])
smartphone_unique['examdate_np'] = pd.to_datetime(smartphone_unique['examdate_np'], errors='coerce')

# Join smartphone data with FHS data based on id.  Retrieve all np exams for patients in the smartphone data
merged_data = pd.merge(smartphone_unique, df_combined, left_on='ptid', right_on='id')

# Calculate the days between the smartphone np examdate and the historical np examdate in the FHS dataset
merged_data['days_smnp_fhsnp'] = (merged_data['examdate_np'] - merged_data['examdate']).abs().dt.days.astype('Int64')



# Keep rows with days between np examdate in the smartphone data and the historical np examdate in the FHS data less than the threshold
# Connect the smartphone data with the historical np exam in the FHS dataset, and retrieve np exam within a predefined timeframe
inference_test = merged_data[merged_data['days_smnp_fhsnp'] <= threshold]




# Get the unique combined keys from the inference_test DataFrame
# Filter out rows from df_combined that have a combined key present in inference_test
training_set = df_combined[~df_combined['combined_key'].isin(inference_test['combined_key'].unique())]



# Inspect and select the np related varaibles
review_df = training_set[training_set['flag'] != "non_review"].copy()
review_cleaned = review_df[review_df['days_flag_np'] <= threshold]

# # Inspect  NP tests in the review set and the inference set
review_missing_summary, review_less_threshold_columns = missing_values_summary(review_cleaned, selected_tests, threshold = 10)
logger.info(f" NP tests with less than 10% missing values in the review set:"
            f"\n {review_less_threshold_columns}")

inference_missing_summary, inference_less_threshold_columns = missing_values_summary(inference_test, selected_tests, threshold = 10)
logger.info(f" NP tests with less than 10% missing values in the inference set:"
            f"\n {inference_less_threshold_columns}")

# Keep the np variables that are less then 10% missing values in both sets
keep_np_tests =[col for col in review_less_threshold_columns if col in inference_less_threshold_columns]
save_columns(keep_np_tests, file_name = "keep_np_tests.txt")
print(keep_np_tests)

# Process the review set
review_result = review_cleaned.loc[review_cleaned[keep_np_tests].notna().any(axis=1)].copy()


# Process the non-review set
non_review_df = training_set.loc[(training_set['flag'] == "non_review")].copy()
non_review_cleaned = non_review_df.loc[(non_review_df['examdate'] > '2021-01-01')].copy()
non_review_result= non_review_cleaned.loc[non_review_cleaned[keep_np_tests].notna().all(axis=1)].copy()


# Save the processed review set,  non-review set, and the inference set
review_result.to_csv(f"dementia_review_preprocessed.csv", index = False)
non_review_result.to_csv(f"non_dementia_review_preprocessed.csv", index = False)
inference_test.to_csv(f"inference_test_preprocessed.csv", index = False)



# Map label
map_to_target = {
    'normal': 0,
    'non_review':0,
    'impairment': 1,
    'mild': 1
}

target_col = 'target'
review_data = map_label(review_result, map_to_target, "flag", target_col)
non_review_data = map_label(non_review_result, map_to_target, "flag", target_col)

if use_non_review:
    df_combined = pd.concat([review_data, non_review_data], ignore_index=True)
    df_final = df_combined.sample(frac=1, random_state=random_state).reset_index(drop=True)
else:
    df_final = review_data
logger.info(f"Target distribution in the combined set\n {count_percentage(df=df_final, col=target_col)}")
logger.info(f"Flag distribution in the combined set\n {count_percentage(df=df_final, col='flag')}")


df_final.to_csv("preprocessed_per_np.csv", index = False)
