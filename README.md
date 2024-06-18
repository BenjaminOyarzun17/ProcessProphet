# Welcome to Process Prophet!


# Integrate our backend into your application
We implemented our backend using a flask backend. Therefore, the backend can also be integrated with 
other frontend projects! If you are a frontend developer/data scientist willing to implement our backend
into your project, **check out the docs**!
[documentation site](https://benjaminoyarzun17.github.io/ProcessProphet/)


# CLI setup and installation
We assume you have docker installed in your machine. Process prophet does support `CUDA`, however this driver must
be configured manually in the `Dockerfile.servercuda`.

## Run without CUDA
First, build both containers and run them using docker compose:
```sh
docker compose up --build
```
alternatively, to run in background: 
```sh
docker compose up --build -d
```
Now, this is a necessary step to run the CLI after doing docker compose: Ideally create another terminal and run: 
```sh
docker-compose exec cli sh
```
this will enter the container's interactive console. Now type:
```sh
python CLI/main.py
```


## Run with CUDA
First, make sure that you have the right CUDA drivers installed, and also edit the CUDA version in the `Dockerfile.servercuda`. 

Run the following command to build and start both containers:
```sh
docker-compose -f docker-compose-cuda.yaml up
```

Now, this is a necessary step to run the CLI after doing docker compose: Ideally create another terminal and run: 
```sh
docker-compose exec cli sh
```
this will enter the container's interactive console. Now type:
```sh
python CLI/main.py
```