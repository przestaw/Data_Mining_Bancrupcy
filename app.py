from loader import loader_bancrupcy
from dataset_analysys import analyse_dataset
from data_input import perf_imputs

if __name__ == "__main__":
    # Load Data from files NOW 1
    dataframes = loader_bancrupcy(1)

    # Analyse sparsity of the data
    # ALREADY DONE : output in ./dataset_analysys
    # analyse_dataset(dataframes)

    dfs_dict = perf_imputs(dataframes)

    # oversampling

    # construct validator? K-fold?

    # solve using random forrest and K-nearest neighbours
