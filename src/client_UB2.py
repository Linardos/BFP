from collections import OrderedDict

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import flwr as fl
import importlib

from models import nets
from data_loader import UB_DataSplit, DataSplit
from tqdm import tqdm
import numpy as np
from pathlib import Path
import pickle

def import_class(name):
    module_name, class_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


HOME_PATH = Path.home()

config_file = 'config.yaml'
with open(config_file) as file:
  CONFIG = yaml.safe_load(file)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = import_class(CONFIG['hyperparameters']['criterion'])

# REMOVE FOR NON-UB CENTERS:
# ub_log_dict = {}
# with open(Path(HOME_PATH / CONFIG['paths']['ub_logs']) / "ub_log.pkl", 'wb') as handle:
#     pickle.dump(ub_log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



def load_data():
    """Load Breast Cancer training and validation set."""
    print('Loading data...')
    dataset = UB_DataSplit(center=CONFIG['center']['UB2'])
    # dataset = DataSplit(0)
    training_loader = DataLoader(dataset.training_set, batch_size=32, shuffle=True)
    validation_loader = DataLoader(dataset.validation_set, batch_size=32, shuffle=True)
    # test_loader = DataLoader(dataset.test_set, batch_size=10, shuffle=True)
    return training_loader, validation_loader #test_loader

# def load_data():
#     """Load CIFAR-10 (training and test set)."""
#     transform = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     )
#     trainset = CIFAR10(".", train=True, download=True, transform=transform)
#     testset = CIFAR10(".", train=False, download=True, transform=transform)
#     trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
#     testloader = DataLoader(testset, batch_size=32)
#     return trainloader, testloader


def train(net, training_loader, epochs, criterion):
    """Train the network on the training set."""
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    losses = []
    cumulative_loss = 0.0
    predictions = []
    print('Training...')
    for _ in range(epochs):
        for i, batch in enumerate(tqdm(training_loader)):
            # images, labels = batch
            # images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE).unsqueeze(1)

            #REMOVE FOR NON-UB CENTERS - take a sample to see the visualize the input to the model
            # if i==0:
            #     with open(Path(HOME_PATH / CONFIG['paths']['ub_logs']) / 'ub_log.pkl', 'rb') as handle:
            #         ub_log_dict = pickle.load(handle)
            #     ub_log_dict['sample_batch'] = images 
            #     with open(Path(HOME_PATH / CONFIG['paths']['ub_logs']) / 'ub_log.pkl', 'wb') as handle:
            #         pickle.dump(ub_log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            cumulative_loss += loss.item()
            loss.backward()
            optimizer.step()

            losses.append(loss)
    #REMOVE FOR NON-UB CENTERS - this needs to be replaced once I figure out how to store predictions in the server
    # with open(Path(HOME_PATH / CONFIG['paths']['ub_logs']) / 'ub_log.pkl', 'rb') as handle:
    #     ub_log_dict = pickle.load(handle)
    # ub_log_dict['losses'] = losses
    # with open(Path(HOME_PATH / CONFIG['paths']['ub_logs']) / 'ub_log.pkl', 'wb') as handle:
    #     pickle.dump(ub_log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    train_results = cumulative_loss #(losses, predictions)
    return train_results

def probabilities_to_labels(predictions : torch.Tensor) -> torch.Tensor:
    """Convert the network's predictions to labels."""
    if predictions.size()[1]==1: # if not one hot encoding
        return torch.round(predictions) #sigmoid outputs
    predictions = predictions.detach().numpy()
    predictions_as_labels = []
    for row in predictions:
        predictions_as_labels.append(np.argmax(row))
    return torch.Tensor(predictions_as_labels)

def test(net, validation_loader, criterion):
    """Validate the network on the entire test set."""
    correct, total, cumulative_loss = 0, 0, 0.0
    predictions = []
    print('Validating...')
    with torch.no_grad():
        for i, batch in enumerate(tqdm(validation_loader)):
            images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE).unsqueeze(1)
            outputs = net(images)
            cumulative_loss += criterion(outputs, labels).item()
            predicted = probabilities_to_labels(outputs.data)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions.append(predicted)
    accuracy = correct / total
    loss = cumulative_loss / total
    print(accuracy)
    test_results = (loss, accuracy) #, predictions)
    return test_results

# Load model and data
# net = nets.ResNet18Classifier(in_ch=3, out_ch=1, linear_ch=512, pretrained=False)
net = nets.SqueezeNetClassifier(in_ch=3, out_ch=1, linear_ch=512, pretrained=True)
net.to(DEVICE)
train_loader, validation_loader = load_data()

class ClassificationClient(fl.client.NumPyClient):
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        results = train(net, train_loader, epochs=1, criterion=CRITERION())
        results = {
            'cumulative_loss': float(results)
        }
        return self.get_parameters(), len(train_loader), results

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        test_results = test(net, validation_loader, criterion=CRITERION())
        loss, accuracy_aggregated = test_results
        test_results = {
            "accuracy":float(accuracy_aggregated),
            # "predictions":predictions
        } #, "predictions": predictions}
        return float(loss), len(validation_loader), test_results 

#fl.client.start_numpy_client("[::]:8080", client=ClassificationClient())
fl.client.start_numpy_client("84.88.186.195:8080", client=ClassificationClient())

# fl.client.start_numpy_client("fe80::d03d:18ff:feb6:d5fa/64", client=ClassificationClient())
# fl.client.start_numpy_client("161.116.4.137:22", client=ClassificationClient())
