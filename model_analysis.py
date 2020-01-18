from collections import OrderedDict
import numpy as np
# To perform kFold Cross Validation
from sklearn.model_selection import KFold
# Models : Random Forrest & K-Neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

from oversample import split_dfs_labels


def prepare_models():
    models_dictionary = OrderedDict()

    # Random Forest Classifiers
    rf_50_trees_gini = RandomForestClassifier(n_estimators=50, criterion='gini')
    rf_10_trees_gini = RandomForestClassifier(n_estimators=10, criterion='gini')
    rf_25_trees_gini = RandomForestClassifier(n_estimators=25, criterion='gini')

    rf_50_trees_entropy = RandomForestClassifier(n_estimators=50, criterion='entropy')
    rf_10_trees_entropy = RandomForestClassifier(n_estimators=10, criterion='entropy')
    rf_25_trees_entropy = RandomForestClassifier(n_estimators=25, criterion='entropy')

    # add to models dictionary
    models_dictionary['Radom Forrest 50 trees gini impurity'] = rf_50_trees_gini
    models_dictionary['Radom Forrest 10 trees gini impurity'] = rf_10_trees_gini
    models_dictionary['Radom Forrest 25 trees gini impurity'] = rf_25_trees_gini

    models_dictionary['Radom Forrest 50 trees information gain'] = rf_50_trees_entropy
    models_dictionary['Radom Forrest 10 trees information gain'] = rf_10_trees_entropy
    models_dictionary['Radom Forrest 25 trees information gain'] = rf_25_trees_entropy

    # K-Neighbors classifiers
    kn_5_uniform_ball_tree = KNeighborsClassifier(nearest_neighbors=5, weights='uniform', algorithm='ball_tree')
    kn_5_distance_ball_tree = KNeighborsClassifier(nearest_neighbors=5, weights='distance', algorithm='ball_tree')

    kn_5_uniform_kd_tree = KNeighborsClassifier(nearest_neighbors=5, weights='uniform', algorithm='kd_tree')
    kn_5_distance_kd_tree = KNeighborsClassifier(nearest_neighbors=5, weights='distance', algorithm='kd_tree')

    kn_5_uniform_brute = KNeighborsClassifier(nearest_neighbors=5, weights='uniform', algorithm='brute')
    kn_5_distance_brute = KNeighborsClassifier(nearest_neighbors=5, weights='distance', algorithm='brute')

    models_dictionary['K nearest neighbors 5 nearest uniform BallTree'] = kn_5_uniform_ball_tree
    models_dictionary['K nearest neighbors 5 nearest distance BallTree'] = kn_5_distance_ball_tree

    models_dictionary['K nearest neighbors 5 nearest uniform KDTree'] = kn_5_uniform_kd_tree
    models_dictionary['K nearest neighbors 5 nearest distance KDTree'] = kn_5_distance_kd_tree

    models_dictionary['K nearest neighbors 5 nearest uniform BruteForce'] = kn_5_uniform_brute
    models_dictionary['K nearest neighbors 5 nearest distance BruteForce'] = kn_5_distance_brute

    return models_dictionary


def prepare_kfold_cv_data(k, x, y, verbose=False):
    y = x.values
    y = y.values
    kf = KFold(n_splits=k, shuffle=False, random_state=42)
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


# perform data modeling
def perform_data_modeling(_models_, _imputers_, verbose=False, k_folds=5):
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
                    print('\t\tDataset: ' + '\033[1m' + str(df_index + 1) + 'year' + '\033[0m')

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
                    _recalls_ = recall_score(y_test, y_test_predicted, average=None)
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

                year_results[str(df_index + 1) + 'year'] = metrics_results

            imputer_results[imputer_name] = year_results

        model_results[model_name] = imputer_results

    return model_results


def prepare_and_do_modeling(df_dict):
    # Prepare models
    models = prepare_models()

    results = perform_data_modeling(models, df_dict, verbose=True, k_folds=5)

    # TODO
    # do ranking,
    # prepare for summary plot

    return None  # TODO
