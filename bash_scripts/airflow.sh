mk dir ariflow
# pwd determines the abs path of the current directory
# airflow_home environment var is set to the current dir abs path
export AIRFLOW_HOME=$(pwd)/airflow

# to inizialize the airflow db go with: 
airflow db init

# if it doesn't work opt for:
# Remove any existing airflow.cfg
rm -f $AIRFLOW_HOME/airflow.cfg
# Reinitialize the database
airflow db init

# creating a user to access the webserver
airflow users create \
    --username admin \
    --firstname Carlo \
    --lastname Airaghi \
    --role Admin \
    --email carlomors96@gmail.com \
    --password ____


# start the web server with 
airflow webserver --port 8080

# in a second terminal lounch the ariflow scheduler (if doing it in parallel bash remember to re-export)
airflow scheduler



