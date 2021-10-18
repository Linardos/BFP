seed = 42  # for reproducibility

# Imports
import os
import yaml
import enum
import copy
import random
import tempfile
import warnings
import multiprocessing
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from collections import OrderedDict

# import visdom
from math import floor, ceil
from pathlib import Path
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
torch.manual_seed(seed)
import torchio as tio
from torchio.transforms import (
    RescaleIntensity,
    RandomElasticDeformation,
    RandomFlip,
    RandomAffine,
    # intensity
    RandomMotion,
    RandomGhosting,
    RandomSpike,
    RandomBiasField,
    RandomBlur,
    RandomNoise,
    RandomSwap,
    RandomAnisotropy,
#     RandomLabelsToImage,
    RandomGamma,
    OneOf,
    CropOrPad,
    ZNormalization,
    HistogramStandardization,
    Compose,
)
from PIL import Image
from src.data_handling.optimam_dataset import OPTIMAMDataset
from src.data_augmentation.breast_density.data.resize_image import *
from torch.utils.data import BatchSampler, RandomSampler 

# Constants
config_file = Path('config.yaml')
with open(config_file) as file:
  config = yaml.safe_load(file)
if config['name']=='sanity_check':
    print("Initiating SANITY CHECK.")
    sanity_check = True
    
# Cropped scans GPU Server
info_csv='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/client_images_screening.csv'
dataset_path='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/images'
output_path = '/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn'
cropped_scans = True
fit_to_breast = True

detection = False
load_max = -1
pathologies = ['mass'] #['mass', 'calcifications', 'suspicious_calcifications', 'architectural_distortion'] # None to select all
# Resize images keeping aspect ratio
rescale_height = 224
rescale_width = 224
plot_images = False

image_ctr = 0

def preprocess_one_image_OPTIMAM(image):
    label = np.single(0) if image.status=='Benign' else np.single(1)
    # status = image.status # ['Benign', 'Malignant', 'Interval Cancer', 'Normal']
    manufacturer = image.manufacturer # ['HOLOGIC, Inc.', 'Philips Digital Mammography Sweden AB', 'GE MEDICAL SYSTEMS', 'Philips Medical Systems', 'SIEMENS']
    view = image.view # MLO_VIEW = ['MLO','LMLO','RMLO', 'LMO', 'ML'] CC_VIEW = ['CC','LCC','RCC', 'XCCL', 'XCCM']
    laterality = image.laterality # L R

    img_pil = Image.open(image.path).convert('RGB')
    img_np = np.array(img_pil)
    scale_size = (rescale_height, rescale_width)
    img_np = np.uint8(img_np) if img_np.dtype != np.uint8 else img_np.copy()
    rescaled_img, scale_factor = imrescale(img_np, scale_size, return_scale=True, backend='pillow')
    image = torch.from_numpy(rescaled_img).permute(2,0,1)

    paddedimg = torch.zeros(3,224,224)
    c,h,w = image.shape
    paddedimg[:,-h:,-w:]=image
    return paddedimg, label

class OPTIMAMDataset_Torch():
    def __init__(self, data_owner):#, manufacturer):
        # Maybe we add the worker here
        optimam_clients = OPTIMAMDataset(info_csv, dataset_path, detection=True, load_max=-1, 
                            cropped_scans=cropped_scans)
        self.clients_selected = optimam_clients.get_clients_by_site(data_owner)
        # self.clients_selected = optimam_clients.get_clients_by_site_and_manufacturer(data_owner, manufacturer)
        self.images = []
        for client in tqdm(self.clients_selected):
            self.images = self.images + client.get_images_by_pathology(['mass'])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # do your handling for a slice object:
            return [preprocess_one_image_OPTIMAM(image) for image in self.images[idx]]
        else:
            # Do your handling for a plain index
            image = self.images[idx]
            return preprocess_one_image_OPTIMAM(image)


class OPTIMAMDataLoader():
    def __init__(self, fold_index, sanity_check=sanity_check):
        all_centers = config['data']['centers']
        test_center = all_centers[fold_index]
        train_centers = [x for i, x in enumerate(all_centers) if i!=fold_index]
        if isinstance(train_centers, list):
            print(f"Train centers: {train_centers}")
            self.training_set, self.validation_set = [], []
            for train_center in train_centers:
                center_dataset = OPTIMAMDataset_Torch(train_center)
                if sanity_check:
                    center_dataset = center_dataset[:10]
                train_size = int(0.9*len(center_dataset))
                self.training_set.append(center_dataset[:train_size])
                self.validation_set.append(center_dataset[train_size:])
        else:
            print("Single train center data loading") # Not adviced
            center_dataset = OPTIMAMDataset_Torch(train_centers)
            train_size = int(0.9*len(center_dataset))
            self.training_set = center_dataset[:train_size]
            self.validation_set = center_dataset[train_size:]

        self.test_set = OPTIMAMDataset_Torch(test_center)

        if sanity_check:
            self.test_set = self.test_set[:3]

    def __len__(self):
        return len(self.training_set) + len(self.test_set)

    # def forward(self, load_test=False):
    #     """This will define the data loaders. There should be a separate one for each center.
    #     """
    #     training_batch_size = config['hyperparameters']['training_batch_size'] 
    #     test_batch_size = config['hyperparameters']['test_batch_size']

    #     if load_test:
    #         print(f"Testing with batch size: {test_batch_size}")
    #         # test_set = self.test_set
    #         test_loader = torch.utils.data.DataLoader(
    #             self.test_set, batch_size=test_batch_size, shuffle=True)
    #         return test_loader

    #     if isinstance(self.training_set, list):
    #         print(f"Training with a batch size of: {training_batch_size}")
    #         # train_loaders, validation_loaders = [], []
    #         for train, val in zip(self.training_set, self.validation_set):
    #             training_loader = torch.utils.data.DataLoader(
    #                 train, batch_size=training_batch_size, shuffle=True)
    #             validation_loader = torch.utils.data.DataLoader(
    #                 val, batch_size=training_batch_size, shuffle=True)
    #             yield training_loader, validation_loader
    #     else:
    #         raise NotImplementedError("You are using a single center for training. This is not implemented and not advised. Please provide a list of training centers")