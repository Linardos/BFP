
# BFP

## Server side:
In config.yaml change the 'docker' variable to True and the 'simulation' variable to False. (lines 4 and 5)
Make sure continue_from_checkpoint is set to False unless you want to load parameters from a pretrained model. (line 60)

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
Make sure you're using CUDA version 11.6
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

docker run -it -v $DATA_PATH:/BFP/dataset -v $PWD/src:/BFP/src -e csv_path=/BFP/dataset/$CSV_FILENAME -e data_loader_type=$DATA_LOADER_TYPE -e dataset_path=/BFP/dataset/ -e server= -e client_log_path=/BFP/src/client_logs -e NVIDIA_VISIBLE_DEVICES=0 --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility registry.gitlab.bsc.es/bfp/fl_breast_mg_classification
```
