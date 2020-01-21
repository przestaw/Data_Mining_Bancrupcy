from loader import loader_bancrupcy
from dataset_analysys import analyse_dataset
from data_input import perf_imputs
from model_analysis import grid_search_rf
from model_analysis import test_knn_vs_rf
import argparse
from oversample import oversample_dfs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predicting bankruptcy from data of Polish companies.')
    parser.add_argument('-t', '--test', help='runs tests and graphs results', required=False, action='store_true')
    parser.add_argument('-g', '--gridsearch', help='run grid search for hyper-parameters for Random Forest Classifier',
                        required=False, action='store_true')
    parser.add_argument('--mean', help='add Mean imputation', required=False, action='store_true', default=False)
    parser.add_argument('--em', help='add EM imputation', required=False, action='store_true', default=False)
    parser.add_argument('--knn', help='add KNN imputation', required=False, action='store_true', default=False)
    parser.add_argument('--mice', help='add MICE imputation', required=False, action='store_true', default=False)
    args = parser.parse_args()

    # Load data from .arff files
    dataframes = loader_bancrupcy(5)

    # Analyse sparsity of the data
    # ALREADY DONE : output in ./dataset_analysys
    # analyse_dataset(dataframes)

    # manage sparsity and imbalance
    if not args.knn and not args.em and not args.mice:
        args.mean = True
    dfs_dict = perf_imputs(dataframes, args.mean, args.em, args.knn, args.mice)
    dfs_oversampled_dict = oversample_dfs(dfs_dict, True)

    # prepare, train and graph model performance
    if args.gridsearch:
        grid_search_rf(dfs_oversampled_dict)
    else:
        test_knn_vs_rf(dfs_oversampled_dict)
