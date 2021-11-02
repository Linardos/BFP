import flwr as fl
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import pickle
import yaml
import os

HOME_PATH = Path.home()
config_file = Path('config.yaml')
with open(config_file) as file:
  CONFIG = yaml.safe_load(file)


log_dict = {'accuracies_aggregated': [],
            'local_loss': []}
PATH_TO_LOG = Path(HOME_PATH / CONFIG['paths']['logs'])
if not os.path.exists(PATH_TO_LOG):
    os.mkdir(PATH_TO_LOG)
with open(PATH_TO_LOG / "log.pkl", 'wb') as handle:
    pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
        with open(PATH_TO_LOG / 'log.pkl', 'rb') as handle:
            log_dict = pickle.load(handle)
        print(log_dict)
        log_dict['local_loss'].append(losses)
        with open(PATH_TO_LOG / "log.pkl", 'wb') as handle:
            pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        """Aggregate model weights using weighted average and store checkpoint"""
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
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
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        print(f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")
        
        with open(Path(HOME_PATH / CONFIG['paths']['logs']) / 'log.pkl', 'rb') as handle:
            log_dict = pickle.load(handle)
        log_dict['accuracies_aggregated'].append(accuracy_aggregated)
        
        with open(Path(HOME_PATH / CONFIG['paths']['logs']) / "log.pkl", 'wb') as handle:
            pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)
# Create strategy and run server
strategy = SaveModelAndMetricsStrategy(
    # (same arguments as FedAvg here)
)

fl.server.start_server(strategy=strategy, server_address="[::]:8000", config={"num_rounds": 30})
