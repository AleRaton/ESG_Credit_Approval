# get the current path
pwd
# create an airflow dir 
mkdir ariflow
# move into it
cd ariflow
# get the docker version
docker --version
# get the docker compose version
docker-compose --version
# get the docker compose yaml file 
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.9.3/docker-compose.yaml'
# remember it could be useful to cut out a few part of the yaml 
# open the airflow dir
cd airflow
# create the other subfolders 
mkdir -p ./dags ./logs ./plugins ./config
echo -e "AIRFLOW_UID=$(id -u)" > .env
# to inizialize the airflow db go with: 
docker compose up airflow-init
# start all services with (use -d if you don't want to see the logs in the terminal)
docker compose up 

# note: if I want to shutdown the container and remove the volume defined in the docker compose yaml file
# docker-compose down -v

# to restart the container after updates of the yaml
docker compose up airflow-init
# start all services with 
docker compose up






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



