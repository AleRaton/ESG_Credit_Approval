import pandas as pd 
import numpy as np

def extract_year(df, column_name):
    # Check if the specified column exists in the DataFrame
    if column_name in df.columns:
        # Extract the year from the column and convert it to an integer
        df[column_name] = df[column_name].str.extract(r'(\d{4})').astype(int)
    return df

def load_clean_data(folder_path, file_name):
    file_path = f"{folder_path}/{file_name}"
    df = pd.read_excel(file_path)
    df=df.T
    new_header = df.iloc[0]                         # first row for the header
    df = df[1:]                                     # take the data less the header row
    df.columns = new_header                         # set the header row as the df header
    df = df.iloc[1:].reset_index(drop=True)         #drop the first row
    df['TICKER'] = file_name.split('.')[0]          #creating a col with the company ticker
    df = df.rename(columns={df.columns[2]: 'Year'})
    df = extract_year(df, 'Year')
    return df

def map_variable_names(df):
    mapping_dict = {
        'Total Revenues': 'Revenues',
        'Net Revenues': 'Revenues',
        'Operating Income': 'Operating_Income',
        'Net Operating Income (NOI)': 'Operating_Income',
        'Net Income to Common': 'Net_Income',
        'Total Current Assets': 'Current_Assets',
        'Total Assets': 'Total_Assets',
        'Total Current Liabilities': 'Current_Liabilities',
        'Total Liabilities': 'Total_Liabilities',
        'Total Equity': 'Equity',
        'Total Capitalization': 'Equity',
        'Total Shareholders Equity': 'Equity',
        'Cash and Equivalents': 'Cash',
        'Net Changes in Cash': 'Cash',
        'Cash From Operations': 'Cash_From_Operations'
    }
    df.rename(columns=mapping_dict, inplace=True)
    return df

def keep_relevant_variables_only(df):
    required_columns = [
        'TICKER',
        'Year',
        'Revenues',
        'Operating_Income',
        'Net_Income',
        'Current_Assets',
        'Total_Assets',
        'Current_Liabilities',
        'Total_Liabilities',
        'Equity',
        'Cash',
        'Cash_From_Operations',
    ]
    
    for col in required_columns:
        if col not in df.columns:
            df[col] = None  # or you can use np.nan if you prefer
    
    df = df[required_columns]
    return df

def clean_and_cast_to_float(df):
    columns_to_cast = [
        'Revenues',
        'Operating_Income',
        'Net_Income',
        'Current_Assets',
        'Total_Assets',
        'Current_Liabilities',
        'Total_Liabilities',
        'Equity',
        'Cash',
        'Cash_From_Operations',
    ]
    
    for col in columns_to_cast:
        if col in df.columns:
            # Replace '-' with None
            df[col] = df[col].replace('—', None)
            # Replace '-' with None
            df[col] = df[col].replace('#N/A Requesting Data...', None)
            # Cast to float
            df[col] = df[col].astype(float)
    
    return df

def get_financial_KPIs(df):
    df['Current_Ratio'] = df['Current_Assets'].div(df['Current_Liabilities'], fill_value=None)
    df['Debt_to_Equity'] = df['Total_Liabilities'].div(df['Equity'], fill_value=None)
    df['Net_Income_to_Assets'] = df['Net_Income'].div(df['Total_Assets'], fill_value=None)
    df['Operating_Margin'] = df['Operating_Income'].div(df['Revenues'], fill_value=None)
    df['Cash_Ratio'] = df['Cash'].div(df['Total_Liabilities'], fill_value=None)
    df['Operating_Cash_Flow_Ratio'] = df['Cash_From_Operations'].div(df['Total_Liabilities'], fill_value=None)
    # Replace inf values with None
    df.replace([np.inf, -np.inf], None, inplace=True)
    df.drop(['Revenues',
            'Operating_Income',
            'Net_Income',
            'Current_Assets',
            'Total_Assets',
            'Current_Liabilities',
            'Total_Liabilities',
            'Equity',
            'Cash',
            'Cash_From_Operations',
             ],axis=1,inplace=True)
    return df

def process_downside_risk(df):
    df = df.T.reset_index()
    new_header = df.iloc[0]                                       # first row for the header
    df = df[1:]                                     # take the data less the header row
    df.columns = new_header                                       # set the header row as the df header
    df.rename(columns={'Anno':'TICKER'},inplace=True)
    df = pd.melt(df, id_vars=['TICKER'], var_name='Year', value_name='downside risk')
    df['Year'] = df['Year'].astype(int)
    return df