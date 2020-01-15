import pandas as pd
# For loading .arff files
from scipy.io import arff


def load_arff_raw_data(N=5):
    return [arff.loadarff('data/' + str(i + 1) + 'year.arff') for i in range(N)]


# Loads the 5 raw .arff files into pandas dataframes
def load_dataframes(N=5):
    return [pd.DataFrame(data_i_year[0]) for data_i_year in load_arff_raw_data(N)]


def loader_bancrupcy(N=5):
    # Load Data from files
    dataframes = load_dataframes(N)

    # Set the column headers from X1 ... X64 and the class label as Y, for all the 5 dataframes.
    cols = ['X' + str(i + 1) for i in range(len(dataframes[0].columns) - 1)]
    cols.append('Y')
    for iter in dataframes:
        iter.columns = cols

    # Convert data to float
    for i in range(5):
        index = 1
        while index <= 63:  # 63 columns
            col_name = dataframes[i].columns[index]
            col = getattr(dataframes[i], col_name)
            dataframes[i][col_name] = col.astype(float)
            index += 1

    # Convert class bancrupcy or not-bancrupcy to integer
    for i in range(len(dataframes)):
        col = getattr(dataframes[i], 'Y')
        dataframes[i]['Y'] = col.astype(int)

    return dataframes
