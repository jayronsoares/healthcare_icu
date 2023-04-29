
"""
In this code, we define each data quality function and a load_data() function to load the dataset from a CSV file using pandas. We also define the main() function to run each data quality check on the dataset and print out the results to the console. The if __name__ == '__main__': block ensures that the main() function is only executed if the script is run directly, and not if it is imported as a module.
"""
import pandas as pd
import numpy as np


def load_data(file_path):
    """
    Load data from a CSV file using pandas.
    """
    df = pd.read_csv(file_path)
    return df


def check_missing_values(df):
    """
    Check for missing values in the dataset using pandas.
    """
    missing_values = df.isnull().sum()
    return missing_values


def check_duplicates(df):
    """
    Check for duplicates in the dataset using pandas.
    """
    duplicates = df.duplicated().sum()
    return duplicates


def check_consistency(df, group_columns, check_column):
    """
    Check for consistency in the dataset using pandas.
    """
    group_data = df.groupby(list(group_columns))[check_column].apply(list).reset_index()
    inconsistent_rows = []
    for i, row in group_data.iterrows():
        if len(set(row[check_column])) > 1:
            inconsistent_rows.append(row)
    return len(inconsistent_rows)


def check_accuracy(df, check_column, min_value, max_value):
    """
    Check for accuracy in the dataset using numpy.
    """
    accuracy_count = np.logical_and(df[check_column] >= min_value, df[check_column] <= max_value).sum()
    accuracy_percent = accuracy_count / len(df) * 100
    return accuracy_count, accuracy_percent


def check_validity(df, check_column, valid_values):
    """
    Check for validity in the dataset using pandas.
    """
    validity_count = df[check_column].isin(valid_values).sum()
    validity_percent = validity_count / len(df) * 100
    return validity_count, validity_percent


def main():
    # Load the dataset
    file_path = 'data.csv'
    df = load_data(file_path)

    # Check for missing values
    missing_values = check_missing_values(df)
    print("Missing Values:\n", missing_values)

    # Check for duplicates
    duplicates = check_duplicates(df)
    print("Number of Duplicates:", duplicates)

    # Check for consistency
    group_columns = ('column1', 'column2')
    check_column = 'column3'
    consistency = check_consistency(df, group_columns, check_column)
    print("Number of Inconsistent Rows:", consistency)

    # Check for accuracy
    check_column = 'column4'
    min_value = 0.0
    max_value = 1.0
    accuracy_count, accuracy_percent = check_accuracy(df, check_column, min_value, max_value)
    print("Number of Accurate Rows:", accuracy_count)
    print("Accuracy Percentage:", accuracy_percent)

    # Check for validity
    check_column = 'column5'
    valid_values = ('value1', 'value2', 'value3')
    validity_count, validity_percent = check_validity(df, check_column, valid_values)
    print("Number of Valid Rows:", validity_count)
    print("Validity Percentage:", validity_percent)


if __name__ == '__main__':
    main()