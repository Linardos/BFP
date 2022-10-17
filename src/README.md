# BFP

To run simulations, you will need to work with the config.yaml file. In simulation mode, the 'docker' variable should be set to False and the 'simulation' variable to True. (lines 4 and 5)

## Running the script:
To run the script you should open multiple windows in tmux, allowing you to run multiple clients at the same time. To do so, you should run the following command:
In one window: python server.py -c 5
In every other window: python client.py 
These will generate two folders in the src/experiment folder, one for the server logs and one for the client logs.
The metrics of interest are stored in the client logs and bear the tag "GMLD" which stands for Global Model Local Data (i.e. the metrics of the model evaluated on the local data of the client without additional training. The AUC should be what interests you).

## Config variables of interest:
aggregator refers to different aggregation options (i.e. FedAvg, FedMedian, FedSmooth) which can be found in the aggregator.py

to change the model architecture, refer to the src/models/nets.py and change the following to your model of choice:

model:
    arch:
        function: models.nets.ResNet18Classifier
