import pandas as pd
import numpy as np
# Library to perform Expectation-Maximization (EM), k-NN and MICE imputations
import impyute as impy
# To perf mean imputation
from sklearn.impute import SimpleImputer
from collections import OrderedDict


# Construct an imputer with strategy as 'mean', to mean-impute along the columns
def perf_mean_imput(dfs_arg):
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    mean_data = [pd.DataFrame(imputer.fit_transform(df)) for df in dfs_arg]
    for i in range(len(dfs_arg)):
        mean_data[i].columns = dfs_arg[i].columns
    return mean_data


# K nearest neighbours
def perf_knn_imput(dfs_arg, n_neighbors):
    # knn_data = [fancyimpute.KNN(k=100, verbose=True).solve(dfs_arg[i])
    knn_data = [impy.fast_knn(dfs_arg[i].values, k=n_neighbors)
                for i in range(len(dfs_arg))]
    return [pd.DataFrame(data=knn_data[i]) for i in range(len(dfs_arg))]


# Expected maximalisation
def perf_em_imput(dfs_arg):
    em_data = [impy.em(dfs_arg[i].values, loops=50, dtype='cont')
               for i in range(len(dfs_arg))]
    return [pd.DataFrame(data=em_data[i]) for i in range(len(dfs_arg))]


# Obtaining the completed features by doing MICE (Multiple Imputation from Chained Equations)
def perf_mice_imput(dfs_arg):
    mice_data = [impy.mice(dfs_arg[i].values, dtype='cont')
                 for i in range(len(dfs_arg))]
    return [pd.DataFrame(data=mice_data[i]) for i in range(len(dfs_arg))]


def set_new_headers(dfs_arg):
    cols = ['X' + str(i + 1) for i in range(len(dfs_arg[0].columns) - 1)]
    cols.append('Y')
    for df in dfs_arg:
        df.columns = cols


def perf_imputs(dfs_arg):
    mean_imputed_dfs = perf_mean_imput(dfs_arg)

    em_imputed_dfs = perf_em_imput(dfs_arg)
    set_new_headers(em_imputed_dfs)
    #knn_imputed_dfs = perf_knn_imput(dfs_arg, 10)
    #set_new_headers(knn_imputed_dfs)
    # mice_imputed_dfs = perf_mice_imput(dfs_arg)
    # set_new_headers(mice_imputed_dfs)

    # Create dictionary for all imputed datasets
    imputed_dfs_dict = OrderedDict()
    imputed_dfs_dict['Mean'] = mean_imputed_dfs
    imputed_dfs_dict['EM'] = em_imputed_dfs
    # imputed_dfs_dict['MICE'] = mice_imputed_dfs

    return imputed_dfs_dict
