
# BFP

## Server side:
In config.yaml change the 'docker' variable to True and the 'simulation' variable to False. (lines 4 and 5)

```bash
center_number = 1
python server.py -c $center_number
```
Where center_number is an integer representing the centers (clients) expected by the script.

## Requirements
- Docker
- Host has NVIDIA GPUs
- Data structure:
	- DATA_PATH/
		- [images_info].csv
		- images/

## Installation

#### Preparing the set-up
```bash
docker pull registry.gitlab.bsc.es/bfp/fl_breast_mg_classification
git clone https://github.com/Linardos/BFP.git 
cd BFP
```
## Run client with CMMD subset. (To run on your own data simply change the DATA_PATH)
CMMD SUBSET:
```bash
DATA_PATH=/home/akis-linardos/Datasets/CMMD_subset
CSV_FILENAME=info.csv
DATA_LOADER_TYPE=general

docker run -it -v $DATA_PATH:/BFP/dataset -v $PWD/src:/BFP/src -e csv_path=/BFP/dataset/$CSV_FILENAME -e data_loader_type=$DATA_LOADER_TYPE -e dataset_path=/BFP/dataset/ -e server=84.88.186.195:8080 -e client_log_path=/BFP/src/client_logs -e NVIDIA_VISIBLE_DEVICES=2 --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility registry.gitlab.bsc.es/bfp/fl_breast_mg_classification

```


## Simulations (Ignore below this point.)

## Run FD client (OPTIMAM)
Defining the environmental variables and running the client.

```bash
DATA_LOADER_TYPE=optimam

docker run -it -v $DATA_PATH:/BFP/dataset -v $PWD/src:/BFP/src -e csv_path=/BFP/dataset/$CSV_FILENAME -e data_loader_type=$DATA_LOADER_TYPE -e dataset_path=/BFP/dataset/images -e server=84.88.186.195:8080 -e client_log_path=/BFP/src/client_logs -e NVIDIA_VISIBLE_DEVICES=2 --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility registry.gitlab.bsc.es/bfp/fl_breast_mg_classification
```

# OPTIMAM stge / jarv center split:
replace CSV_FILENAME=client_images_screening.csv with: CSV_FILENAME=jarv_info.csv or CSV_FILENAME=stge_info.csv 


## For clients using BCDR, InBreast or CMMD datasets:

InBreast:
```bash
DATA_LOADER_TYPE=inbreast

docker run -it -v $DATA_PATH:/BFP/dataset -v $PWD/src:/BFP/src -e csv_path=/BFP/dataset/$CSV_FILENAME -e data_loader_type=$DATA_LOADER_TYPE -e dataset_path=/BFP/dataset/AllPNG_cropped -e server=84.88.186.195:8080 -e client_log_path=/BFP/src/client_logs -e NVIDIA_VISIBLE_DEVICES=2 --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility registry.gitlab.bsc.es/bfp/fl_breast_mg_classification

```

BCDR:
```bash
DATA_LOADER_TYPE=bcdr

docker run -it -v $DATA_PATH:/BFP/dataset -v $PWD/src:/BFP/src -e csv_path=/BFP/dataset/ -e data_loader_type=$DATA_LOADER_TYPE -e dataset_path=/BFP/dataset/ -e server=84.88.186.195:8080 -e client_log_path=/BFP/src/client_logs -e NVIDIA_VISIBLE_DEVICES=2 --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility registry.gitlab.bsc.es/bfp/fl_breast_mg_classification

```

CMMD:
```bash
DATA_PATH=/home/akis-linardos/Datasets/CMMD
CSV_FILENAME=info.csv
DATA_LOADER_TYPE=cmmd

docker run -it -v $DATA_PATH:/BFP/dataset -v $PWD/src:/BFP/src -e csv_path=/BFP/dataset/$CSV_FILENAME -e data_loader_type=$DATA_LOADER_TYPE -e dataset_path=/BFP/dataset/ -e server=84.88.186.195:8080 -e client_log_path=/BFP/src/client_logs -e NVIDIA_VISIBLE_DEVICES=2 --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility registry.gitlab.bsc.es/bfp/fl_breast_mg_classification

```



