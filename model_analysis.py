from collections import OrderedDict
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
from tensorflow.python.ops.gen_clustering_ops import nearest_neighbors


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
