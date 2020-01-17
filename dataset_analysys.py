import matplotlib.pyplot as plt
import missingno as msno


# Get Clean dataframes by dropping all the rows which have missing values
def drop_nan_rows(dataframes_arg, print_stats=False):
    clean_dataframes = [df.dropna(axis=0, how='any') for df in dataframes_arg]
    if print_stats:
        for i in range(len(dataframes_arg)):
            print(str(i + 1) + 'year:', 'Original Length=', len(dataframes_arg[i]), '\tCleaned Length=',
                  len(clean_dataframes[i]), '\tMissing Data=', len(dataframes_arg[i]) - len(clean_dataframes[i]))

    return clean_dataframes


# generate the sparsity matrix (figure) for all the dataframes
def generate_sparsity_matrix(dataframes_arg, print=False):
    for i in range(5):
        missing_df_i = dataframes_arg[i].columns[dataframes_arg[i].isnull().any()].tolist()

        ax0 = fig = msno.matrix(dataframes_arg[i][missing_df_i], figsize=(20, 5))
        if print:
            plt.show()

        plt.savefig('year' + str(i) + '.png')


def check_data_imbalance(dataframes_arg):
    for i in range(len(dataframes_arg)):
        print('Dataset: ' + str(i + 1) + 'year')
        print(dataframes_arg[i].groupby('Y').size())
        minority_percent = (dataframes_arg[i]['Y'].tolist().count(1) / len(dataframes_arg[i]['Y'].tolist())) * 100
        print('Minority (label 1) percentage: ' + str(minority_percent) + '%')
        print('-' * 64)


def analyse_dataset(dataframes_arg):
    # drop_nan_rows(dataframes_arg, True)
    # generate_sparsity_matrix(dataframes_arg, True)
    check_data_imbalance(dataframes_arg)
