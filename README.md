
# BFP

## Requirements
- Docker
- Host has NVIDIA GPUs
- Data structure:
	- DATA_PATH/
		- [images_info].csv
		- images/

## Installation

#### Option 1: Pulling image from dockerhub

#### Option 2: Building image
```bash
git clone https://github.com/Linardos/BFP.git
cd docker
docker build -t bfp_docker .
```
 


## Run FD client

```bash
DATA_PATH=/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed
CSV_FILENAME=client_images_screening.csv
IMAGES_FOLDER=images
DATA_LOADER_TYPE=optimam

cd BFP
docker run -it -v $DATA_PATH:/BFP/dataset -v $PWD/src:/BFP/src -e csv_path=/BFP/dataset/$CSV_FILENAME -e dataset_path=/BFP/dataset/$IMAGES_FOLDER -e data_loader_type=$DATA_LOADER_TYPE -e server=84.88.186.195:8080 bfp_docker
```

## For clients using BCDR or InBreast:

InBreast:
```bash
DATA_PATH=/home/lidia-garrucho/datasets/INBREAST/
CSV_FILENAME=INbreast_updated_cropped_breast.csv
IMAGES_FOLDER=AllPNG_cropped
DATA_LOADER_TYPE=inbreast
```

BCDR:
```bash
DATA_PATH=/home/lidia/Datasets/BCDR/cropped/
CSV_FILENAME=None
DATA_LOADER_TYPE=bcdr

docker run -it -v $DATA_PATH:/BFP/dataset -v $PWD/src:/BFP/src -e csv_path=/BFP/dataset/$CSV_FILENAME -e dataset_path=/BFP/dataset -e data_loader_type=$DATA_LOADER_TYPE -e server=84.88.186.195:8080 bfp_docker
```