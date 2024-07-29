from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import glob
from utils import data_prep_functions as prep

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'financial_data_processing',
    default_args=default_args,
    description='A simple data processing DAG',
    schedule_interval=None,  # Set to None to run on demand
)

folder_path = "data/financial_data"

def load_clean_financial_data(file_name):
    df = prep.load_clean_data(folder_path, file_name)
    df = prep.map_variable_names(df)
    df = prep.keep_relevant_variables_only(df)
    df = prep.clean_and_cast_to_float(df)
    df = prep.get_financial_KPIs(df)
    return df

def process_financial_data():
    file_list = glob.glob(f"{folder_path}/*.xlsx")
    companies_data = []
    for file_path in file_list:
        file_name = file_path.split('/')[-1]
        df = load_clean_financial_data(file_name)
        companies_data.append(df)
    df_financial = pd.concat(companies_data)
    df_financial.to_csv('/tmp/df_financial.csv', index=False)

def process_esg_data():
    esg_overall = pd.read_excel("data/esg/esg_overall.xlsx")
    esg_environmental = pd.read_excel("data/esg/esg_environmental.xlsx")
    esg_social = pd.read_excel("data/esg/esg_social.xlsx")
    esg_governance = pd.read_excel("data/esg/esg_governance.xlsx")
    
    esg_overall = pd.melt(esg_overall, id_vars=['TICKER', 'Settore'], var_name='Year', value_name='esg overall')
    esg_environmental = pd.melt(esg_environmental, id_vars=['TICKER', 'Settore'], var_name='Year', value_name='esg environmental')
    esg_social = pd.melt(esg_social, id_vars=['TICKER', 'Settore'], var_name='Year', value_name='esg social')
    esg_governance = pd.melt(esg_governance, id_vars=['TICKER', 'Settore'], var_name='Year', value_name='esg governance')
    
    dfs = [esg_overall, esg_environmental, esg_social, esg_governance]
    df_esg = dfs[0]
    for df in dfs[1:]:
        df_esg = pd.merge(df_esg, df, on=['TICKER', 'Year', 'Settore'], how='left')
    df_esg.sort_values(['TICKER', 'Year'])
    df_esg.to_csv('/tmp/df_esg.csv', index=False)

def process_downside_risk():
    df_downside_risk = pd.read_excel('data/risk/downside_risk.xlsx')
    df_downside_risk = prep.process_downside_risk(df_downside_risk)
    df_downside_risk.sort_values(['TICKER', 'Year'])
    df_downside_risk.to_csv('/tmp/df_downside_risk.csv', index=False)

def merge_data():
    df_financial = pd.read_csv('/tmp/df_financial.csv')
    df_esg = pd.read_csv('/tmp/df_esg.csv')
    df_downside_risk = pd.read_csv('/tmp/df_downside_risk.csv')
    
    df = pd.merge(df_financial, df_esg, on=['TICKER', 'Year'], how='left')
    df = pd.merge(df, df_downside_risk, on=['TICKER', 'Year'], how='left')
    df.to_csv('data/processed_data/prepared_data.csv', index=False)

with dag:
    task_process_financial_data = PythonOperator(
        task_id='process_financial_data',
        python_callable=process_financial_data
    )

    task_process_esg_data = PythonOperator(
        task_id='process_esg_data',
        python_callable=process_esg_data
    )

    task_process_downside_risk = PythonOperator(
        task_id='process_downside_risk',
        python_callable=process_downside_risk
    )

    task_merge_data = PythonOperator(
        task_id='merge_data',
        python_callable=merge_data
    )

    task_process_financial_data >> task_process_esg_data >> task_process_downside_risk >> task_merge_data
