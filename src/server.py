from typing import List, Tuple, Optional
import os
from collections import OrderedDict
from datetime import datetime
import time
import shutil
import glob

import flwr as fl
import numpy as np
import torch
import pickle
import yaml
import importlib
import argparse
from pathlib import Path

from models import nets
import aggregator

parser = argparse.ArgumentParser()
parser.add_argument("-c","--center_number", type=int, help="Center Number", default=5)
args = parser.parse_args()
# print(args.center_number)

HOME_PATH = Path.home()
config_file = Path('config.yaml')
with open(config_file) as file:
  CONFIG = yaml.safe_load(file)

os.environ['server'] = f"161.116.4.132:{CONFIG['port']}"
DEVICE = torch.device(CONFIG['device'])
def import_class(name):
    module_name, class_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

### ====== Make Experiment Folder ====== ###

path_to_experiments = HOME_PATH / Path(CONFIG['paths']['experiments']) # Server script doesn't use Docker so it's fine.

if not os.path.exists(path_to_experiments):
    os.mkdir(path_to_experiments)

experiment_name = CONFIG['name']
if experiment_name[0:4].isdigit():
    # note that if you want to continue or overwrite an experiment you should use the full name including the digit
    new_experiment_name = experiment_name 
else:
    # add a new tag to create a new experiment
    max_tag = 0
    for file in path_to_experiments.iterdir():
        if str(file.name)[0:4].isdigit():
            if max_tag < int(str(file.name)[0:4]):
                max_tag = int(str(file.name)[0:4])
    tag = str(max_tag+1).zfill(4)
    new_experiment_name = tag + '_' + experiment_name

    
    '''
    To avoid cluttering the experiments folder when dealing with errors, 
    this will make sure not to create a new tag for a duplicate file name that's been created within the last 10 minutes
    '''
    possible_last_file = str(max_tag).zfill(4) + '_' +experiment_name 
    possible_last_file = path_to_experiments.joinpath(possible_last_file)
    if possible_last_file.exists():
        timestamp = datetime.fromtimestamp(possible_last_file.stat().st_mtime)
        now = datetime.now()
        if timestamp.hour == now.hour and now.minute-timestamp.minute<20:
            new_experiment_name = possible_last_file # overwrite in this case
            
    
PATH_TO_EXPERIMENT = path_to_experiments.joinpath(new_experiment_name)
print(f"Generating experiment {new_experiment_name}")

# else:
#     if not config['model']['continue']:
#         print(f"Deleting {MODEL_STORAGE}")
#         for f in MODEL_STORAGE.iterdir():
#             f.unlink() #empty directory for new model states. 
#make a copy of config file in new experiment to keep track of parameters.

if not os.path.exists(PATH_TO_EXPERIMENT):
    os.makedirs(PATH_TO_EXPERIMENT)
shutil.copy(config_file, PATH_TO_EXPERIMENT)
shutil.copy('server.py', PATH_TO_EXPERIMENT)
shutil.copy('data_loader.py', PATH_TO_EXPERIMENT)
#\\ ====== Make Experiment Folder ====== #\\


### ====== Log ====== ###
log_dict = {'accuracies_aggregated': [],
            'total_val_loss': [],
            'time_spent': []}
with open(PATH_TO_EXPERIMENT / 'log.pkl', 'wb') as handle:
    pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#\\ ====== Log ====== #\\

# net = nets.ResNet101Classifier(in_ch=3, out_ch=1, pretrained=False).to(DEVICE)
# net = import_class(CONFIG['model']['arch']['function'])
INITIAL_TIME = time.time()

Model = import_class(CONFIG['model']['arch']['function'])
if CONFIG['model']['arch']['args']:
    net = Model(**CONFIG['model']['arch']['args'])
else:
    net = Model()

### ====== Load previous Checkpoint (Under Development) ====== ###
# def load_parameters_from_disk():
if CONFIG['paths']['continue_from_checkpoint']:
    # import Net
    list_of_files = [fname for fname in glob.glob(CONFIG['paths']['continue_from_checkpoint']+"/model_round_*")]
    latest_round_file = max(list_of_files, key=os.path.getctime)
    print("Loading pre-trained model from: ", latest_round_file)
    state_dict = torch.load(latest_round_file)
    net.load_state_dict(state_dict)
#\\ ====== Load previous Checkpoint (Under Development) ====== #\\


### ====== Center Dropout (Under Development) ====== ###
# class CDCriterion(fl.server.criterion.Criterion):
#     def __init__(self, criterion, dropout_prob):
#         super().__init__()
#         self.criterion = criterion
#         self.dropout_prob = dropout_prob

#     def select():
        

# class CenterDropoutClientManager(fl.server.client_manager.SimpleClientManager):
#     def sample(
#         self,
#         num_clients: int,
#         min_num_clients: Optional[int] = None,
#         criterion: Optional[Criterion] = None,
#     ) -> List[ClientProxy]:
#         """Apply Center Dropout."""

#         # Block until at least num_clients are connected.
#         if min_num_clients is None:
#             min_num_clients = num_clients
#         self.wait_for(min_num_clients)
#         # Sample clients which meet the criterion
#         available_cids = list(self.clients)
#         if criterion is not None:
#             available_cids = [
#                 cid for cid in available_cids if criterion.select(self.clients[cid])
#             ]
#         sampled_cids = random.sample(available_cids, num_clients)
#         return [self.clients[cid] for cid in sampled_cids]
#\\ ====== Center Dropout (Under Development) ====== #\\


def import_class(name):
    module_name, class_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


### ====== Strategy for Checkpointing and Metrics ====== ###
class SaveModelAndMetricsStrategy(import_class(CONFIG['strategy']['aggregator'])):
    #
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]], # FitRes is like EvaluateRes and has a metrics key 
        failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:

        losses = [r.metrics["cumulative_loss"] for _, r in results]
        print(f"Length results is {len(losses)}")

        # Store metrics into log_dict
        with open(PATH_TO_EXPERIMENT / 'log.pkl', 'rb') as handle:
            log_dict = pickle.load(handle)
        print(log_dict)
        log_dict['total_val_loss'].append(losses)
        log_dict['time_spent'].append(time.time() - INITIAL_TIME)
        
        with open(PATH_TO_EXPERIMENT / "log.pkl", 'wb') as handle:
            pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if CONFIG['model']['arch']['args']:
            net = Model(**CONFIG['model']['arch']['args'])
        else:
            net = Model()

        """Aggregate model weights using weighted average and store checkpoint"""
        aggregated_parameters_tuple = super().aggregate_fit(rnd, results, failures)
        aggregated_parameters, _ = aggregated_parameters_tuple
        # log_dict['aggregated_parameters']=aggregated_parameters
        
        if aggregated_parameters is not None:
            print(f"Saving round {rnd} aggregated_parameters...")
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_weights: List[np.ndarray] = fl.common.parameters_to_weights(aggregated_parameters)
            
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_weights)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
            # Save the model
            torch.save(net.state_dict(), PATH_TO_EXPERIMENT / f"model_round_{rnd}.pth")
            
        return aggregated_parameters_tuple 

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Weigh accuracy of each client by number of examples used
        
        # print([r.metrics.keys() for _, r in results])
        # accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        accuracies = [r.metrics["accuracy"] for _, r in results]
        print(accuracies)
        # print([r.metrics["predictions"] for _, r in results])
        # examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / len(accuracies)
        print(f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")
        
        with open(PATH_TO_EXPERIMENT / 'log.pkl', 'rb') as handle:
            log_dict = pickle.load(handle)
        log_dict['accuracies_aggregated'].append(accuracy_aggregated)
        
        with open(PATH_TO_EXPERIMENT / "log.pkl", 'wb') as handle:
            pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)
#\\ ====== Strategy for Checkpointing and Metrics ====== #\\

### ====== Configure file to set parameters for federated rounds ====== ###
def fit_config(rnd: int):
    """Return training configuration dict for each round.
    """
    if CONFIG['hyperparameters']['epochs_strategy'] == 'dynamic_epochs':
        if rnd < int(CONFIG['hyperparameters']['federated_rounds'])/3:
            local_epochs = 1
        elif rnd < int(CONFIG['hyperparameters']['federated_rounds'])*2/3:
            local_epochs = 2
        else:
            local_epochs = 3
        config = {
            # "batch_size": 32,
            "local_epochs": local_epochs if rnd < int(CONFIG['hyperparameters']['federated_rounds']) \
                else int(CONFIG['hyperparameters']['epochs_last_round']),
            "round_number": rnd
        }
    else: #Tuning on last round
        config = {
            # "batch_size": 32,
            "local_epochs": int(CONFIG['hyperparameters']['epochs_per_round']) if rnd < int(CONFIG['hyperparameters']['federated_rounds']) \
                else int(CONFIG['hyperparameters']['epochs_last_round']),
            "round_number": rnd
        }

    return config
#\\ ====== Configure file to set parameters for federated rounds ====== #\\

### ====== Define strategy and initiate server ====== ###

# if CONFIG['paths']['continue_from_checkpoint']:
#     strategy = SaveModelAndMetricsStrategy(
#         # (same arguments as FedAvg here)
#         min_available_clients = args.center_number,
#         min_fit_clients = args.center_number,
#         min_eval_clients = args.center_number,
#         # min_available_clients = CONFIG['strategy']['min_available_clients'],
#         # min_fit_clients = CONFIG['strategy']['min_fit_clients'],
#         # min_eval_clients = CONFIG['strategy']['min_eval_clients'],
#         fraction_fit = CONFIG['strategy']['CD_P'],
#         on_fit_config_fn = fit_config,
#         initial_parameters=load_parameters_from_disk() # if CONFIG['paths']['continue_from_checkpoint'] else None
#     )
# else:
strategy = SaveModelAndMetricsStrategy(
    # (same arguments as FedAvg here)
    min_available_clients = args.center_number,
    min_fit_clients = args.center_number,
    min_eval_clients = args.center_number,
    # min_available_clients = CONFIG['strategy']['min_available_clients'],
    # min_fit_clients = CONFIG['strategy']['min_fit_clients'],
    # min_eval_clients = CONFIG['strategy']['min_eval_clients'],
    fraction_fit = CONFIG['strategy']['CD_P'],
    on_fit_config_fn = fit_config
)

fl.server.start_server(strategy=strategy, server_address=f"[::]:{CONFIG['port']}", config={"num_rounds": CONFIG['hyperparameters']['federated_rounds']})
#\\ ====== Define strategy and initiate server ====== #\\