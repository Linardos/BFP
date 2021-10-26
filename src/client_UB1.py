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

def import_class(name):
    module_name, class_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
    
config_file = 'config.yaml'
with open(config_file) as file:
  train_conf = yaml.safe_load(file)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = import_class(train_conf['hyperparameters']['criterion'])


def load_data():
    """Load training and validation set."""
    dataset = UB_DataSplit(center=train_conf['center']['UB1'])
    # dataset = DataSplit(0)
    training_loader = DataLoader(dataset.training_set, batch_size=32, shuffle=True)
    validation_loader = DataLoader(dataset.validation_set, batch_size=32, shuffle=True)
    # test_loader = DataLoader(dataset.test_set, batch_size=10, shuffle=True)
    return training_loader, validation_loader #test_loader

def train(net, training_loader, epochs, criterion):
    """Train the network on the training set."""
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for i, batch in enumerate(tqdm(training_loader)):
            images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

def test(net, validation_loader, criterion):
    """Validate the network on the entire test set."""
    correct, total, loss = 0, 0, 0.0
    predictions = []
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE).unsqueeze(1)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions.append(predicted)
    accuracy = correct / total
    return loss, (accuracy, predictions)


# Load model and data
# net = nets.ResNet18Classifier(in_ch=3, out_ch=10, linear_ch=512, pretrained=True)
net = nets.SqueezeNetClassifier(in_ch=3, out_ch=1, linear_ch=512, pretrained=True)
net.to(DEVICE)
train_loader, validation_loader = load_data()

class BreastClassificationClient(fl.client.NumPyClient):
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, train_loader, epochs=1, criterion=CRITERION())
        return self.get_parameters(), len(train_loader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, results_tuple = test(net, validation_loader, criterion=CRITERION())
        accuracy, predictions = results_tuple
        results = {"accuracy":float(accuracy)} #, "predictions": predictions}
        return float(loss), len(validation_loader), results 

fl.client.start_numpy_client("[::]:8080", client=BreastClassificationClient())
# fl.client.start_numpy_client("fe80::d03d:18ff:feb6:d5fa/64", client=BreastClassificationClient())
# fl.client.start_numpy_client("161.116.4.137:22", client=BreastClassificationClient())
