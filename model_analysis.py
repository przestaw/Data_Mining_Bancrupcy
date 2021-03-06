from collections import OrderedDict
from copy import copy

import numpy as np
# To perform kFold Cross Validation
from sklearn.model_selection import KFold
# Models : Random Forrest & K-Neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

from oversample import split_dfs_labels


class ModelsDict:
    models = OrderedDict()

    def add_rf(self, _n_estimators, _criterion, _max_features='auto', _max_depth=None):
        cl = RandomForestClassifier(n_estimators=_n_estimators, criterion=_criterion, max_features=_max_features,
                                    max_depth=_max_depth)
        key = 'RF/'+str(_n_estimators)+'/'+_criterion
        self.models[key] = cl

    def add_knn(self, _n_neighbors, _weights, _algorithm):
        cl = KNeighborsClassifier(n_neighbors=_n_neighbors, weights=_weights, algorithm=_algorithm)
        key = 'KNN/' + str(_n_neighbors) + '/' + _weights + '/' + _algorithm
        self.models[key] = cl


def prepare_kfold_cv_data(k, x, y, verbose=False):
    x = x.values
    y = y.values
    kf = KFold(n_splits=k, shuffle=False)
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for train_index, test_index in kf.split(x):
        x_train.append(x[train_index])
        y_train.append(y[train_index])
        x_test.append(x[test_index])
        y_test.append(y[test_index])
    return x_train, y_train, x_test, y_test


def fit_grid_random_forest(dataframes, verbose=True, k_folds=5):
    if verbose:
        print("-" * 120, "\n", "Grid search for hyper-parameters for " + '\033[1m' + "Random Forest" +
              '\033[0m' + " Classifier, using "+ '\033[1m' + "Mean "+ '\033[0m' + "imputing method:")
    imputer_results = OrderedDict()

    # grid search
    prm_grid = [{'n_estimators': [1, 10, 100], 'criterion': ['gini', 'entropy'], 'max_features': ['auto', 'log2', 16],
                 'max_depth': [10, 100, 1000, None]}]
    cl = RandomForestClassifier()
    grid = GridSearchCV(verbose=verbose, estimator=cl, param_grid=prm_grid, n_jobs=-1)
    # call the split_dfs_labels function to get a list of features and labels for all the dataframes
    feature_dfs, label_dfs = split_dfs_labels(dataframes)

    # list of classifiers for each year
    cls = []

    # Iterate over dataframe_list individually
    for df_index in range(len(dataframes)):
        if verbose:
            print('\t\tDataset: ' + '\033[1m' + str(df_index + 1) + ' year' + '\033[0m')

        # Fit the model and
        grid = grid.fit(feature_dfs[df_index], label_dfs[df_index])
        cls.append(copy(grid.best_estimator_))
        print('\t\tBest score: ', grid.best_score_)
    return cls


def fit_custom_models(_models_dict, _imputers_, verbose=True, k_folds=5):
    _models_ = _models_dict.models
    model_results = OrderedDict()

    # Iterate over the models
    for model_name, clf in _models_.items():
        if verbose:
            print("-" * 120, "\n", "Model: " + '\033[1m' + model_name + '\033[0m' + " Classifier")
        imputer_results = OrderedDict()


        # Iterate over the different imputed_data mechanisms like : (Mean, k-NN, EM, MICE)
        for imputer_name, dataframes_list in _imputers_.items():
            if verbose:
                print('\tImputer Technique: ' + '\033[1m' + imputer_name + '\033[0m')

            # call the split_dfs_labels function to get a list of features and labels for all the dataframes
            feature_dfs, label_dfs = split_dfs_labels(dataframes_list)

            year_results = OrderedDict()

            # Iterate over dataframe_list individually
            for df_index in range(len(dataframes_list)):
                if verbose:
                    print('\t\tDataset: ' + '\033[1m' + str(df_index + 1) + ' year' + '\033[0m')

                # Calling the 'prepare_kfold_cv_data' returns lists of features and labels
                # for train and test sets respectively.
                # The number of items in the list is equal to k_folds
                x_train_list, y_train_list, x_test_list, y_test_list = \
                    prepare_kfold_cv_data(k_folds,
                                          feature_dfs[df_index],
                                          label_dfs[df_index],
                                          verbose)

                metrics_results = OrderedDict()
                accuracy_list = np.zeros([k_folds])
                precision_list = np.zeros([k_folds, 2])
                recall_list = np.zeros([k_folds, 2])
                TN_list = np.zeros([k_folds])
                FP_list = np.zeros([k_folds])
                FN_list = np.zeros([k_folds])
                TP_list = np.zeros([k_folds])

                # Iterate over all the k-folds
                for k_index in range(k_folds):
                    x_train = x_train_list[k_index]
                    y_train = y_train_list[k_index]
                    x_test = x_test_list[k_index]
                    y_test = y_test_list[k_index]

                    # Fit the model and
                    clf = clf.fit(x_train, y_train)
                    y_test_predicted = clf.predict(x_test)

                    # code for calculating accuracy
                    _accuracy_ = accuracy_score(y_test, y_test_predicted, normalize=True)
                    accuracy_list[k_index] = _accuracy_

                    # code for calculating recall
                    _recalls_ = recall_score(y_test, y_test_predicted, average=None, zero_division=1)
                    recall_list[k_index] = _recalls_

                    # code for calculating precision
                    _precisions_ = precision_score(y_test, y_test_predicted, average=None)
                    precision_list[k_index] = _precisions_

                    # code for calculating confusion matrix
                    _confusion_matrix_ = confusion_matrix(y_test, y_test_predicted)
                    TN_list[k_index] = _confusion_matrix_[0][0]
                    FP_list[k_index] = _confusion_matrix_[0][1]
                    FN_list[k_index] = _confusion_matrix_[1][0]
                    TP_list[k_index] = _confusion_matrix_[1][1]

                # creating a metrics dictionary
                metrics_results['Accuracy'] = np.mean(accuracy_list)
                metrics_results['Precisions'] = np.mean(precision_list, axis=0)
                metrics_results['Recalls'] = np.mean(recall_list, axis=0)
                metrics_results['TN'] = np.mean(TN_list)
                metrics_results['FP'] = np.mean(FP_list)
                metrics_results['FN'] = np.mean(FN_list)
                metrics_results['TP'] = np.mean(TP_list)

                if verbose:
                    print('\t\t\tAccuracy:', metrics_results['Accuracy'])
                    print('\t\t\tPrecision:', metrics_results['Precisions'])
                    print('\t\t\tRecall:', metrics_results['Recalls'])

                year_results[df_index + 1] = metrics_results

            imputer_results[imputer_name] = year_results

        model_results[model_name] = imputer_results

    return model_results


def plot_results(results, _title):
    for model_name, imputers in results.items():
        for imputer_name, years in imputers.items():
            x = []
            y = []
            for year, metrics in years.items():
                x.append(year)
                y.append(metrics['Accuracy']*100)
            plt.plot(x, y, label=model_name + " " + imputer_name, marker='.')
    plt.xlabel("Year")
    plt.title(_title)
    plt.ylabel("Accuracy [%]")
    plt.legend()
    plt.show()


# already found out that Random Forest is way better -> now find the best RF Classifier parameters
def grid_search_rf(df_dict):
    best_models = fit_grid_random_forest(df_dict['Mean'], verbose=True, k_folds=5)
    i = 1
    for model in best_models:
        print("Best model for year", i)
        print(model.get_params())
        i += 1


# generate plots from .doc, proves which technique is more accurate
def test_knn_vs_rf(df_dict):
    # KNN testing
    # test how many neighbours to consider as nearest
    models = ModelsDict()
    models.add_knn(1, 'uniform', 'kd_tree')
    models.add_knn(10, 'uniform', 'kd_tree')
    models.add_knn(25, 'uniform', 'kd_tree')
    results = fit_custom_models(models, df_dict, verbose=True, k_folds=5)
    plot_results(results, "Varied number of nearest neighbours")

    # check for distance-based weights
    models.models.clear()
    models.add_knn(1, 'distance', 'kd_tree')
    models.add_knn(10, 'distance', 'kd_tree')
    models.add_knn(25, 'distance', 'kd_tree')
    results = fit_custom_models(models, df_dict, verbose=True, k_folds=5)
    plot_results(results, "Varied number of nearest neighbours (distance weight)")

    # check which distance gives better results for n_neighbors = 1
    models.models.clear()
    models.add_knn(1, 'uniform', 'kd_tree')
    models.add_knn(1, 'distance', 'kd_tree')
    results = fit_custom_models(models, df_dict, verbose=True, k_folds=5)
    plot_results(results, "Comparing uniform vs distance neighbour weights")

    # Random Forest Testing
    # test amount of trees with gini
    models.models.clear()
    models.add_rf(1, 'gini')
    models.add_rf(10, 'gini')
    models.add_rf(100, 'gini')
    results = fit_custom_models(models, df_dict, verbose=True, k_folds=5)
    plot_results(results, "Varied amount of trees [gini]")

    # test the same with entropy
    models.models.clear()
    models.add_rf(1, 'entropy')
    models.add_rf(10, 'entropy')
    models.add_rf(100, 'entropy')
    results = fit_custom_models(models, df_dict, verbose=True, k_folds=5)
    plot_results(results, "Varied amount of trees [entropy]")

    # entropy vs gini with 100 trees
    models.models.clear()
    models.add_rf(100, 'gini')
    models.add_rf(100, 'entropy')
    results = fit_custom_models(models, df_dict, verbose=True, k_folds=5)
    plot_results(results, "entropy vs gini")

    # best KNN vs best Random Forest comparison
    models = ModelsDict()
    models.add_knn(1, 'distance', 'kd_tree')
    models.add_rf(100, 'gini')
    results = fit_custom_models(models, df_dict, verbose=True, k_folds=5)
    plot_results(results, "best KNN vs best RF")

