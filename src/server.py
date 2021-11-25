import flwr as fl
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import pickle
import yaml
import os
from models import nets
from collections import OrderedDict
import torch
from datetime import datetime
import time
import shutil

HOME_PATH = Path.home()
config_file = Path('config.yaml')
with open(config_file) as file:
  CONFIG = yaml.safe_load(file)

def import_class(name):
    module_name, class_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

log_dict = {'accuracies_aggregated': [],
            'total_val_loss': [],
            'time_spent': []}
path_to_experiments = HOME_PATH / Path(CONFIG['paths']['experiments']) # Without Docker
# path_to_experiments = Path("/") / Path(CONFIG['paths']['experiments']) # With Docker

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


with open(PATH_TO_EXPERIMENT / 'log.pkl', 'wb') as handle:
    pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

net = nets.ResNet101Classifier(in_ch=3, out_ch=1, pretrained=False)
# net = import_class(CONFIG['model']['arch']['function'])
INITIAL_TIME = time.time()
class SaveModelAndMetricsStrategy(fl.server.strategy.FedAvg):
    #
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]], # FitRes is like EvaluateRes and has a metrics key 
        failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:

        # print([r.metrics.keys() for _, r in results])
        losses = [r.metrics["cumulative_loss"] for _, r in results]
        print(f"Length results is {len(losses)}")
        with open(PATH_TO_EXPERIMENT / 'log.pkl', 'rb') as handle:
            log_dict = pickle.load(handle)
        print(log_dict)
        log_dict['total_val_loss'].append(losses)
        log_dict['time_spent'].append(time.time() - INITIAL_TIME)

        """Aggregate model weights using weighted average and store checkpoint"""
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            print(f"Saving round {rnd} aggregated_weights...")
            # params_dict = zip(net.state_dict().keys(), aggregated_weights)
            # state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            # torch.save(state_dict, f"round-{rnd}-weights.pt")
            # Save aggregated_weights
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights) # How to save as state_dict PyTorch
        # log_dict['model_weights']=aggregated_weights
        with open(PATH_TO_EXPERIMENT / "log.pkl", 'wb') as handle:
            pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return aggregated_weights

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
# Create strategy and run server
strategy = SaveModelAndMetricsStrategy(
    # (same arguments as FedAvg here)
)

fl.server.start_server(strategy=strategy, server_address="[::]:8080", config={"num_rounds": CONFIG['hyperparameters']['federated_rounds']})
