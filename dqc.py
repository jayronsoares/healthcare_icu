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
    return df.isnull().sum()

def check_duplicates(df):
    """
    Check for duplicates in the dataset using pandas.
    """
    return df.duplicated().sum()

def check_consistency(df, group_columns, check_column):
    """
    Check for consistency in the dataset using pandas.
    """
    group_data = df.groupby(list(group_columns))[check_column].apply(list).reset_index()
    return group_data.apply(lambda row: len(set(row[check_column])) > 1, axis=1).sum()

def check_accuracy_validity(df, column):
    """
    Check for accuracy and validity in the dataset using numpy and pandas.
    """
    min_value, max_value = df[column].min(), df[column].max()
    valid_values = df[column].unique()
    
    accuracy_count = np.logical_and(df[column] >= min_value, df[column] <= max_value).sum()
    accuracy_percent = accuracy_count / len(df) * 100
    
    validity_count = df[column].isin(valid_values).sum()
    validity_percent = validity_count / len(df) * 100
    
    return accuracy_count, accuracy_percent, validity_count, validity_percent

def check_cardinality(df, column):
    """
    Check if a column has high or low cardinality.
    """
    cardinality = df[column].nunique()
    high_cardinality = cardinality > len(df) / 2
    low_cardinality = cardinality < 0.01 * len(df)
    return high_cardinality, low_cardinality

def print_data_quality_results(results_dict):
    """
    Print data quality results in a formatted way.
    """
    for column, result in results_dict.items():
        print(f"Column: {column}")
        print("Missing Values:", result["Missing Values"])
        print("Duplicates:", result["Duplicates"])
        print("High Cardinality:", result["High Cardinality"])
        print("Low Cardinality:", result["Low Cardinality"])
        print("Consistency Issues:", result["Consistency Issues"])
        print("Accuracy Count:", result["Accuracy Count"])
        print("Accuracy Percentage:", result["Accuracy Percentage"])
        print("Validity Count:", result["Validity Count"])
        print("Validity Percentage:", result["Validity Percentage"])
        print("\n")

def main():
    # Load the dataset
    file_path = "Suicide data.csv"
    df = load_data(file_path)

    # Dictionary to store data quality results
    results_dict = {}

    # Check for data quality for each column in the DataFrame
    for column in df.columns:
        results_dict[column] = {
            "Missing Values": check_missing_values(df[column]),
            "Duplicates": check_duplicates(df[column]),
            "High Cardinality": check_cardinality(df, column)[0],
            "Low Cardinality": check_cardinality(df, column)[1],
            "Consistency Issues": check_consistency(df, group_columns=df.drop(column, axis=1).columns[:2], check_column=column),
        }
        
        # Get accuracy and validity results
        accuracy_count, accuracy_percent, validity_count, validity_percent = check_accuracy_validity(df, column)
        results_dict[column]["Accuracy Count"] = accuracy_count
        results_dict[column]["Accuracy Percentage"] = accuracy_percent
        results_dict[column]["Validity Count"] = validity_count
        results_dict[column]["Validity Percentage"] = validity_percent

    # Print data quality results
    print_data_quality_results(results_dict)

if __name__ == '__main__':
    main()
