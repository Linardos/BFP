
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
cd BFP/docker
docker build -t bfp_docker .
```
 


## Run FD client

```bash
DATA_PATH=/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed
CSV_FILENAME=client_images_screening.csv

cd BFP
docker run -it -v $DATA_PATH:/BFP/dataset -v src:/BFP/src -e csv_path=/BFP/dataset/$CSV_FILENAME -e dataset_path=/BFP/dataset/images -e server=84.88.186.195:8080 bfp_docker
```

