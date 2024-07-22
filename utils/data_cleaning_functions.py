import pandas as pd

def impute_missing_data_by_group(df,group):
    """
    Impute missing data in a DataFrame by replacing missing values with the average
    of the corresponding feature for the population of the same group.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    
    Returns:
    pandas.DataFrame: The DataFrame with missing values imputed.
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    imputed_df = df.copy()
    
    # Iterate over each column in the DataFrame
    for col in imputed_df.columns:
        # Skip the 'group' column
        if col == group:
            continue
        
        # Calculate the mean value for each feature within each group
        group_means = imputed_df.groupby(group)[col].transform('mean')
        
        # Fill in the missing values with the group-specific mean
        imputed_df[col] = imputed_df[col].fillna(group_means)
    
    return imputed_df