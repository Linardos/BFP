
# BFP

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

## Run FD client
Defining the environmental variables and running the client.

```bash
DATA_PATH=/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed
CSV_FILENAME=client_images_screening.csv
IMAGES_FOLDER=images
DATA_LOADER_TYPE=optimam

docker run -it -v $DATA_PATH:/BFP/dataset -v $PWD/src:/BFP/src -e csv_path=/BFP/dataset/$CSV_FILENAME -e data_loader_type=$DATA_LOADER_TYPE -e dataset_path=/BFP/dataset/images -e server=84.88.186.195:8080 -e client_log_path=/BFP/src/client_logs -e NVIDIA_VISIBLE_DEVICES=2 --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility registry.gitlab.bsc.es/bfp/fl_breast_mg_classification
```

## For clients using BCDR, InBreast or CMMD datasets:

InBreast:
```bash
DATA_PATH=/home/lidia-garrucho/datasets/INBREAST
CSV_FILENAME=INbreast_updated_cropped_breast.csv
IMAGES_FOLDER=AllPNG_cropped
DATA_LOADER_TYPE=inbreast

docker run -it -v $DATA_PATH:/BFP/dataset -v $PWD/src:/BFP/src -e csv_path=/BFP/dataset/$CSV_FILENAME -e data_loader_type=$DATA_LOADER_TYPE -e dataset_path=/BFP/dataset/AllPNG_cropped -e server=84.88.186.195:8080 -e client_log_path=/BFP/src/client_logs -e NVIDIA_VISIBLE_DEVICES=2 --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility registry.gitlab.bsc.es/bfp/fl_breast_mg_classification

```

BCDR:
```bash
DATA_PATH=/home/lidia-garrucho/datasets/BCDR
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

CMMD SUBSET:
```bash
DATA_PATH=/home/akis-linardos/Datasets/CMMD_subset
CSV_FILENAME=info.csv
DATA_LOADER_TYPE=cmmd

docker run -it -v $DATA_PATH:/BFP/dataset -v $PWD/src:/BFP/src -e csv_path=/BFP/dataset/$CSV_FILENAME -e data_loader_type=$DATA_LOADER_TYPE -e dataset_path=/BFP/dataset/ -e server=84.88.186.195:8080 -e client_log_path=/BFP/src/client_logs -e NVIDIA_VISIBLE_DEVICES=2 --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility registry.gitlab.bsc.es/bfp/fl_breast_mg_classification
```
