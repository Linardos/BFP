import flwr as fl
import numpy as np
from typing import List, Tuple, Optional

class SaveModelAndMetricsStrategy(fl.server.strategy.FedAvg):
    #
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

    def aggregate_evaluate(
        self,
        rnd: int,
        results, #: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Weigh accuracy of each client by number of examples used
        
        # print([r.metrics.keys() for _, r in results])
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        # print([r.metrics["predictions"] for _, r in results])
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        print(f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)
# Create strategy and run server
strategy = SaveModelAndMetricsStrategy(
    # (same arguments as FedAvg here)
)

fl.server.start_server(strategy=strategy, config={"num_rounds": 3})
