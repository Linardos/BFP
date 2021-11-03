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
from collections import OrderedDict

from math import floor, ceil
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
torch.manual_seed(seed)
from PIL import Image
from src.data_augmentation.breast_density.data.resize_image import *
from src.preprocessing.histogram_standardization import apply_hist_stand_landmarks
from src.data_handling.mmg_detection_datasets import *

from torch.utils.data import BatchSampler, RandomSampler 

# Constants

HOME_PATH = Path.home()
config_file = Path('config.yaml')
with open(config_file) as file:
  CONFIG = yaml.safe_load(file)
if CONFIG['name']=='sanity_check':
    print("Initiating SANITY CHECK.")
    sanity_check = True
    
# Cropped scans GPU Server
# csv_path=CONFIG['paths']['csv_path']
# dataset_path=CONFIG['paths']['datapath'] 
cropped_to_breast = True
fit_to_breast = True

detection = False
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
    # view = image.view # MLO_VIEW = ['MLO','LMLO','RMLO', 'LMO', 'ML'] CC_VIEW = ['CC','LCC','RCC', 'XCCL', 'XCCM']
    # laterality = image.laterality # L R

    img_pil = Image.open(image.path).convert('RGB')
    img_np = np.array(img_pil)
    scale_size = (rescale_height, rescale_width)
    img_np = np.uint8(img_np) if img_np.dtype != np.uint8 else img_np.copy()
    rescaled_img, scale_factor = imrescale(img_np, scale_size, return_scale=True, backend='pillow')
    image = torch.from_numpy(rescaled_img).permute(2,0,1)
    
    # Histogram Matching 
    landmarks_values = torch.load(HOME_PATH / CONFIG['paths']['landmarks'])
    apply_hist_stand_landmarks(image, landmarks_values)

    paddedimg = torch.zeros(3,224,224)
    c,h,w = image.shape
    paddedimg[:,-h:,-w:]=image
    return paddedimg, label

class ALLDataset(): # Should work for any center
    def __init__(self, dataset_path, csv_path, mode='train', load_max=1000, center=None): 
        subjects = OPTIMAMDataset(csv_path, dataset_path, detection=False, load_max=-1, 
                            cropped_to_breast=cropped_to_breast) # we should be able to load any dataset with this
        
        subjects_selected = {}
        if center!=None:
            total_subjects = subjects.get_images_by_site(center)
        else:
            # General case
            subjects_selected['benign'] = subjects.get_clients_by_status('Benign')[:load_max] #Note that clients means subjects here.
            subjects_selected['malignant'] = subjects.get_clients_by_status('Malignant')[:load_max]
            subjects_selected['normal'] = subjects.get_clients_by_status('Normal')[:load_max]
            total_subjects = subjects_selected['benign'] + subjects_selected['malignant'] + subjects_selected['normal']
        random.shuffle(total_subjects) 
        # Data Split
        training_subjects = total_subjects[:int(len(total_subjects)*0.8)]
        validation_subjects = total_subjects[int(len(total_subjects)*0.8):int(len(total_subjects)*0.9)]
        test_subjects = total_subjects[int(len(total_subjects)*0.9):]

        def extract_images(subjects):
            images=[]
            for subject in tqdm(subjects):
                for study in subject:
                    for image in study:
                        images.append(image)
            return images

        if mode == 'train':
            self.images = extract_images(training_subjects)
        elif mode == 'val':
            self.images = extract_images(validation_subjects) 
        elif mode == 'test':
            self.images = extract_images(test_subjects)

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
