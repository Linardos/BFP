from collections import OrderedDict

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from torchvision.datasets import CIFAR10
import flwr as fl
from datetime import datetime
import importlib
import os

from models import nets
from data_loader import ALLDataset
from tqdm import tqdm
import numpy as np
from pathlib import Path
import pickle
# from sklearn import roc_auc_score
# import argparse


def import_class(name):
    module_name, class_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

HOME_PATH = Path.home()

config_file = 'config.yaml'
with open(config_file) as file:
  CONFIG = yaml.safe_load(file)

CSV_PATH = os.environ['csv_path']
DATASET_PATH = os.environ['dataset_path']
# Before running each client locally, you need to set the environment variable client_log_path to a unique value for each worker.

### IMPORTANT!!! When running out of docker, run this on terminal first:
# export client_log_path=/home/akis-linardos/BFP/src/client_logs/c1 <- should be unique for each worker (c1, c2, etc)
LOG_PATH = Path(os.environ['client_log_path']) # This is to store client results
os.makedirs(LOG_PATH, exist_ok=True)

# Global Model Local Data : GMLD
# Local Model Local Data : LMLD
# Global Model Aggregated Metrics : GMAM

log_dict = {'local_loss':{0:[]}, 'local_val_loss':{0:[]}, 'local_accuracy':[], 'local_sensitivity':[], 'local_specificity':[], 'local_val_predictions':[],
            'GMLD_accuracy':[], 'GMLD_true_positives':[],'GMLD_false_positives':[],'GMLD_false_negatives':[],'GMLD_true_negatives':[],
            'LMLD_train_accuracy':[], 'LMLD_val_accuracy':[],
            'LMLD_train_true_positives':[], 'LMLD_train_false_positives':[], 'LMLD_train_false_negatives':[], 'LMLD_train_true_negatives':[],
            'LMLD_val_true_positives':[], 'LMLD_val_false_positives':[], 'LMLD_val_false_negatives':[], 'LMLD_val_true_negatives':[],}
with open(LOG_PATH / "log.pkl", 'wb') as handle:
    pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# SERVER=os.environ['server']
# SERVER = "161.116.4.132:8080" # server without docker at BCN-AIM cluster
# SERVER= os.getenv('server',"[::]:8080")
SERVER= os.getenv('server',"161.116.4.132:8080") 
DATA_LOADER_TYPE= os.getenv('data_loader_type',"optimam") #env variable data_loader if not given default to optimam type dataloading

# Docker ip is: 172.17.0.3
print(f'Here dataset path {DATASET_PATH}')
print(f'Here csv path {CSV_PATH}')

# parser = argparse.ArgumentParser(description='Process some integers.')
# # parser.add_argument('-c', '--csv', help='path to csv', default=CONFIG['paths']['csv_path'])
# # parser.add_argument('-d', '--dataset', help='path to dataset', default=CONFIG['paths']['dataset_path'])
# parser.add_argument('--center', help='use only when you have multi-center data', default=None)

# args = parser.parse_args()

DEVICE = torch.device(CONFIG['device']) #if torch.cuda.is_available() else "cpu")
CRITERION = import_class(CONFIG['hyperparameters']['criterion'])

def load_data():
    """Load Breast Cancer training and validation set."""
    print('Loading data...')
    training_loader = DataLoader(ALLDataset(DATASET_PATH, CSV_PATH, mode='train', data_loader_type=DATA_LOADER_TYPE, load_max=CONFIG['data']['load_max']), batch_size=CONFIG['hyperparameters']['batch_size'])
    validation_loader = DataLoader(ALLDataset(DATASET_PATH, CSV_PATH, mode='val', data_loader_type=DATA_LOADER_TYPE, load_max=CONFIG['data']['load_max']), batch_size=CONFIG['hyperparameters']['batch_size'])
    test_loader = DataLoader(ALLDataset(DATASET_PATH, CSV_PATH, mode='test', data_loader_type=DATA_LOADER_TYPE, load_max=CONFIG['data']['load_max']), batch_size=CONFIG['hyperparameters']['batch_size'])
    print(len(training_loader))
    return training_loader, validation_loader #test_loader

def train(net, training_loader, validation_loader, criterion):
    """Train the network on the training set."""
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    losses = []
    cumulative_loss = 0.0
    predictions = []
    print('Training...')
    with open(LOG_PATH / 'log.pkl', 'rb') as handle:
        log_dict = pickle.load(handle)
    current_round_num=list(log_dict['local_loss'].keys())[-1]
    next_round_num=current_round_num+1
    log_dict['local_loss'][next_round_num]=[] # A key for the next round is generated. Final round will always remain empty
    for _ in range(CONFIG['hyperparameters']['epochs_per_round']):
        for i, batch in enumerate(tqdm(training_loader)):
            images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            cumulative_loss += loss.item()
            loss.backward()
            optimizer.step()

        log_dict['local_loss'][current_round_num].append(loss.item())

    # FOR SANITY CHECK, REMOVE LATER:
    correct, total, cumulative_loss = 0, 0, 0.0
    false_positive, false_negative, true_positive, true_negative = 0, 0, 0, 0
    for i, batch in enumerate(tqdm(training_loader)):
        images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE).unsqueeze(1)
        outputs = net(images)
        loss = criterion(outputs, labels).item()
        #
        cumulative_loss += criterion(outputs, labels).item()
        predicted = probabilities_to_labels(outputs.data)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        false_positive += ((predicted == 1) & (labels == 0)).sum().item()
        false_negative += ((predicted == 0) & (labels == 1)).sum().item()
        true_positive += ((predicted == 1) & (labels == 1)).sum().item()
        true_negative += ((predicted == 0) & (labels == 0)).sum().item()

    accuracy = correct / total
    # sensitivity = true_positive.sum().item() / (true_positive.sum().item() + false_negative.sum().item())
    # specificity = true_negative.sum().item() / (true_negative.sum().item() + false_positive.sum().item())
    # AUC = roc_auc_score(labels.detach().numpy(), outputs.detach().numpy())
    log_dict['LMLD_train_accuracy'].append(accuracy)
    # Store everything!
    log_dict['LMLD_train_true_positives'].append(true_positive)
    log_dict['LMLD_train_false_positives'].append(false_positive)
    log_dict['LMLD_train_true_negatives'].append(true_negative)
    log_dict['LMLD_train_false_negatives'].append(false_negative)

    
    correct, total, cumulative_loss = 0, 0, 0.0
    false_positive, false_negative, true_positive, true_negative = 0, 0, 0, 0
    for i, batch in enumerate(tqdm(validation_loader)):
        images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE).unsqueeze(1)
        outputs = net(images)
        loss = criterion(outputs, labels).item()
        #
        cumulative_loss += criterion(outputs, labels).item()
        predicted = probabilities_to_labels(outputs.data)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        false_positive += ((predicted == 1) & (labels == 0)).sum().item()
        false_negative += ((predicted == 0) & (labels == 1)).sum().item()
        true_positive += ((predicted == 1) & (labels == 1)).sum().item()
        true_negative += ((predicted == 0) & (labels == 0)).sum().item()

    accuracy = correct / total
    # sensitivity = true_positive.sum().item() / (true_positive.sum().item() + false_negative.sum().item())
    # specificity = true_negative.sum().item() / (true_negative.sum().item() + false_positive.sum().item())
    # AUC = roc_auc_score(labels.detach().numpy(), outputs.detach().numpy())
    log_dict['LMLD_val_accuracy'].append(accuracy)
    # Store everything!
    log_dict['LMLD_val_true_positives'].append(true_positive)
    log_dict['LMLD_val_false_positives'].append(false_positive)
    log_dict['LMLD_val_true_negatives'].append(true_negative)
    log_dict['LMLD_val_false_negatives'].append(false_negative)
    ####  

    print(log_dict)
    with open(LOG_PATH / "log.pkl", 'wb') as handle:
        pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
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
    false_positive, false_negative, true_positive, true_negative = 0, 0, 0, 0
    predictions, val_losses = [], []
    print('Validating...')
    with open(LOG_PATH / 'log.pkl', 'rb') as handle:
        log_dict = pickle.load(handle)
    current_round_num=list(log_dict['local_val_loss'].keys())[-1]
    next_round_num=current_round_num+1
    log_dict['local_val_loss'][next_round_num]=[] # A key for the next round is generated. Final round will always remain empty
    with torch.no_grad():
        for i, batch in enumerate(tqdm(validation_loader)):
            images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE).unsqueeze(1)
            outputs = net(images)
            loss = criterion(outputs, labels).item()
            # 
            cumulative_loss += criterion(outputs, labels).item()
            predicted = probabilities_to_labels(outputs.data)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            false_positive += ((predicted == 1) & (labels == 0)).sum().item()
            false_negative += ((predicted == 0) & (labels == 1)).sum().item()
            true_positive += ((predicted == 1) & (labels == 1)).sum().item()
            true_negative += ((predicted == 0) & (labels == 0)).sum().item()
            # predictions.append(predicted)
            val_losses.append(loss)
    val_loss=sum(val_losses)/len(val_losses)
    log_dict['local_val_loss'][current_round_num]=val_loss
    accuracy = correct / total
    # sensitivity = true_positive.sum().item() / (true_positive.sum().item() + false_negative.sum().item())
    # specificity = true_negative.sum().item() / (true_negative.sum().item() + false_positive.sum().item())
    # AUC = roc_auc_score(labels.detach().numpy(), outputs.detach().numpy())
    log_dict['GMLD_accuracy'].append(accuracy)
    # log_dict['local_sensitivity'].append(sensitivity)
    # log_dict['local_specificity'].append(specificity)
    # Store everything!
    log_dict['GMLD_true_positives'].append(true_positive)
    log_dict['GMLD_false_positives'].append(false_positive)
    log_dict['GMLD_true_negatives'].append(true_negative)
    log_dict['GMLD_false_negatives'].append(false_negative)

    loss = cumulative_loss / total
    print(accuracy)
    print(log_dict)
    with open(LOG_PATH / "log.pkl", 'wb') as handle:
        pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # import pdb; pdb.set_trace()
    # test_results = (loss, accuracy, bytes(predictions))
    test_results = (loss, accuracy)
    return test_results



class ClassificationClient(fl.client.NumPyClient):
    def __init__(self):
        super(ClassificationClient, self).__init__()
        self.net = nets.ResNet101Classifier(in_ch=3, out_ch=1, pretrained=False)
        self.net.to(DEVICE)
        self.train_loader, self.validation_loader = load_data()

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        results = train(self.net, self.train_loader, self.validation_loader, criterion=CRITERION()) # validation loader for sanity check only
        results = {
            'cumulative_loss': float(results)
        }
        return self.get_parameters(), len(self.train_loader), results

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        test_results = test(self.net, self.validation_loader, criterion=CRITERION())
        loss, accuracy_aggregated = test_results
        test_results = {
            "accuracy":float(accuracy_aggregated),
            "loss":float(loss)
            # "predictions":predictions
        }
        return float(loss), len(self.validation_loader), test_results 

# fl.client.start_numpy_client("[::]:8080", client=ClassificationClient())
# fl.client.start_numpy_client("161.116.4.132", client=ClassificationClient())
#fl.client.start_numpy_client("84.88.186.195:8080", client=ClassificationClient())

fl.client.start_numpy_client(SERVER, client=ClassificationClient())

# cd BFP, then:
# docker run -it -v $DATA_PATH:/BFP/dataset -v $PWD/src:/BFP/src -e csv_path=/BFP/dataset/$CSV_FILENAME -e dataset_path=/BFP/dataset/$IMAGES_FOLDER -e data_loader_type=$DATA_LOADER_TYPE -e server=161.116.4.137:8080 bfp_docker
# or just:
# docker run -it -v $DATA_PATH:/BFP/dataset -v /home/akis-linardos/BFP/src:/BFP/src -e csv_path=/BFP/dataset/$CSV_FILENAME -e dataset_path=/BFP/dataset/images -e data_loader_type=optimam -e server=161.116.4.137:8080 bfp_docker
