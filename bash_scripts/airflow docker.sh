# get the current path
pwd
# create an airflow dir 
mkdir airflow
# move into it
cd airflow
# get the docker version
docker --version
# get the docker compose version
docker-compose --version
# get the docker compose yaml file 
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.9.3/docker-compose.yaml'
# remember it could be useful to cut out a few part of the yaml 
# create the other subfolders 
mkdir -p ./dags ./logs ./plugins ./config
echo -e "AIRFLOW_UID=$(id -u)" > .env
# to inizialize the airflow db go with: 
docker compose up airflow-init
# (ues it also to restart the container after updates of the yaml)
# start all services with (use -d if you don't want to see the logs in the terminal)
docker compose up 

# note: if I want to shutdown the container and remove the volume defined in the docker compose yaml file
# docker-compose down -v

# ensure all the necessary containers are up and running
docker ps








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



# in a second terminal lounch the airflow scheduler (if doing it in parallel bash remember to re-export)
airflow scheduler



