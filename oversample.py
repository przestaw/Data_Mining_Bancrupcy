# Library imbalanced-learn to deal with the data imbalance. To use SMOTE oversampling
from imblearn.over_sampling import SMOTE
import pandas as pd
from collections import OrderedDict
from collections import Counter
# reuse code
from data_input import set_new_headers


# Split the features and labels into separate dataframes for all the original dataframes
def split_dfs_labels(dfs):
    feature_dfs = [dfs[i].iloc[:, 0:64] for i in range(len(dfs))]
    label_dfs = [dfs[i].iloc[:, 64] for i in range(len(dfs))]
    return feature_dfs, label_dfs


# Performs the SMOTE oversampling for given dataframes.
def smote_oversampling(dfs, verbose=False):
    smote = SMOTE(random_state=42, k_neighbors=10)
    # Split the features and labels for each dataframe
    feat_dfs, label_dfs = split_dfs_labels(dfs)
    resampled_feat_arr = []
    resampled_label_arr = []
    for i in range(len(dfs)):
        if verbose:
            print('Dataset: ' + str(i + 1) + 'year:')
        if verbose:
            print('Original dataset shape {}'.format(Counter(label_dfs[i])))
        dfi_features_res, dfi_label_res = smote.fit_sample(feat_dfs[i], label_dfs[i])
        if verbose:
            print('Resampled dataset shape {}\n'.format(Counter(dfi_label_res)))
        # Append the resampled feature and label arrays of ith dataframe to their respective list of arrays
        resampled_feat_arr.append(dfi_features_res)
        resampled_label_arr.append(dfi_label_res)
    return resampled_feat_arr, resampled_label_arr


# Utility Function to convert the arrays of features and labels to pandas dataframes, and then join them.
# Also re-assign the columns headers.
def arr_to_dfs(feature_arr, label_arr):
    dfs = []
    for i in range(len(feature_arr)):
        feature_df = pd.DataFrame(data=feature_arr[i])
        label_df = pd.DataFrame(data=label_arr[i])
        # Must set the column header for label_df,
        # otherwise it wont join with feature_df, as columns overlap (with col names '0')
        label_df.columns = ['Y']
        dfs.append(feature_df.join(label_df))
    # re-assign the column headers for features and labels
    set_new_headers(dfs)
    return dfs


# Perform SMOTE oversampling on all dataframes
def oversample_dfs(df_dict, verbose=False):
    # Make return dictionary object
    ret_dict = OrderedDict()
    for key, dfs in df_dict.items():
        if verbose:
            print('SMOTE Oversampling for ' + key + ' imputed dataframes\n')

        # feature and label arrays
        smote_f_arr, smote_l_arr = smote_oversampling(dfs, verbose=verbose)
        # make dataframes again
        oversampled_dfs = arr_to_dfs(smote_f_arr, smote_l_arr)
        # make dictionary gain
        ret_dict[key] = oversampled_dfs

        if verbose:
            print()
    return ret_dict
