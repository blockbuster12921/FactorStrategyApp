# Setup

## Prerequisites
- Install Docker

## Run web application with uwsgi/nginx web server
- Navigate to the root folder of the application (the folder containing docker-compose.yml)
- Run the web server:
    docker-compose build
    docker-compose up
- In a web browser, browse to http://localhost:8080

## Run web application with debug web server
- Ensure that Docker is installed
- Navigate to the root folder of the application (the folder containing docker-compose.yml)
- Run the debug web server:
    docker-compose -f docker-compose.yml -f docker-compose.debug.yml build
    docker-compose -f docker-compose.yml -f docker-compose.debug.yml up
- In a web browser, browse to http://localhost:5003

## Run jupyter notebook
- Run the following command:
    docker-compose -f docker-compose.jupyter.yml up
- Browse to http://localhost:8890/

## Run unit tests
- Without the web server running, run the following command:
    docker-compose run --rm flask_app "python -m unittest"
- With the web server running, run the following command:
    docker-compose exec flask_app python -m unittest

# Usage









 





