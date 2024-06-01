# SPP-process-discovery
it is recommended to install docker desktop for container management. 
# running with docker compose
this commands builds both containers and runs them:
```sh
docker compose up --build
```
alternatively, to run in background: 
```sh
docker compose up --build -d
```
Now, this is a necessary step to run the CLI after doing docker compose. Note that building the 
server container usually takes a while due to the large amount of dependencies!
ideally create another terminal and run: 
```sh
docker-compose exec cli sh
```
this will enter the container's interactive console. Now type
```sh
python CLI/main.py
```
this will start the CLI.


# Environment variable
inside the CLI directory there is a `.env` file. Create inside this file an environment variable
```sh
SERVER_NAME=localhost
```
this will be changed automatically by docker compose.

## building server
(just use for testing purposes)
```sh
docker buildx build -f Dockerfile.server -t ppserver .
```

## running cli container 
(just use for testing purposes)
```sh
sudo docker run -v ./projects:/projects -it ppcli
```
this grants access to the container's terminal



# to run the server locally (Achtung!!)
```sh
python -m server.server
```
this ensures to run the server as a MODULE. otherwise the imports will not work.


# using mkdocs
run the following command to run the documentation on local host:
```bash
mkdocs serve
```
the mkdocs config file is `mkdocs.yml`. The markdown templates are in `/docs`. 


# installing requirements using a virtual environment: 
run the following commands: 
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```