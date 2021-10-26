from collections import OrderedDict

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import flwr as fl

from models import nets
from data_loader import UB_DataSplit 
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_file = 'config.yaml'
with open(config_file) as file:
  train_conf = yaml.safe_load(file)

def load_data():
    """Load CIFAR-10 (training and test set)."""
    # transform = transforms.Compose(
    # [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )
    # trainset = CIFAR10(".", train=True, download=True, transform=transform)
    # testset = CIFAR10(".", train=False, download=True, transform=transform)
    # for train_set, val_set in zip(UB_DataSplit.training_set, UB_DataSplit.validation_set):
    dataset = UB_DataSplit(0,center=config['center']['UB1'])
    training_loader = DataLoader(dataset.training_set, batch_size=32, shuffle=True)
    validation_loader = DataLoader(dataset.validation_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset.test_set, batch_size=10, shuffle=True)
    return training_loader, validation_loader #test_loader

def train(net, training_loader, epochs, criterion):
    """Train the network on the training set."""
    #criterion = torch.nn.CrossEntropyLoss()
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
    #criterion = torch.nn.CrossEntropyLoss()
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
        train(net, train_loader, epochs=1, criterion=train_conf['hyperparameters']['criterion'])
        return self.get_parameters(), len(train_loader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        import pdb; pdb.set_trace()
        loss, results_tuple = test(net, validation_loader, criterion=train_conf['hyperparameters']['criterion'])
        accuracy, predictions = results_tuple
        results = {"accuracy":float(accuracy)} #, "predictions": predictions}
        return float(loss), len(validation_loader), results 

#fl.client.start_numpy_client("[::]:8080", client=BreastClassificationClient())
#fl.client.start_numpy_client("10.32.7.16:8080", client=CifarClient())
#fl.client.start_numpy_client("fl.server.bsc.es", client=CifarClient())
fl.client.start_numpy_client("84.88.186.195:8080", client=BreastClassificationClient())

# fl.client.start_numpy_client("fe80::d03d:18ff:feb6:d5fa/64", client=BreastClassificationClient())
# fl.client.start_numpy_client("161.116.4.137:22", client=BreastClassificationClient())
