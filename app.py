from loader import loader_bancrupcy

if __name__ == "__main__":
    # Load Data from files
    dataframes = loader_bancrupcy(5)

    # Analyse sparsity of the data
    # ALREADY DONE : output in ./dataset_analysys
    # analyse_dataset(dataframes)
