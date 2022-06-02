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
import sys
sys.path.append('/BFP')

from math import floor, ceil
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
torch.manual_seed(seed)
from PIL import Image
import pydicom as dicom

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
    
pathologies = ['mass'] #['mass', 'calcifications', 'suspicious_calcifications', 'architectural_distortion'] # None to select all
# Resize images keeping aspect ratio
rescale_height = 224
rescale_width = 224

image_ctr = 0

LANDMARKS = os.environ['landmarks']# CONFIG['paths']['landmarks']

# Handle DICOMs:
# "/home/kaisar/Datasets/InBreast/AllDICOMs"

def crop_MG(arr): # remove zeroes side

    mask = arr != 0
    n = mask.ndim
    dims = range(n)
    slices = [None]*n

    for i in dims:
        mask_i = mask.any(tuple([*dims[:i], *dims[i+1:]]))
        slices[i] = (mask_i.argmax(), len(mask_i) - mask_i[::-1].argmax())

    return arr[[slice(*s) for s in slices]]

def preprocess_one_image_OPTIMAM(image): # Read as nifti without saving
    if image.status=='Malignant' or image.status=='Malign':
        label = np.single(1)
    elif image.status=='Benign': 
        label = np.single(0)
    else: #Add normal eventually
        raiseError("Unknown status: {}".format(image.status))
    # label = np.single(1) if image.status=='Malignant' else np.single(0)
    # status = image.status # ['Benign', 'Malignant', 'Interval Cancer', 'Normal']
    manufacturer = image.manufacturer # ['HOLOGIC, Inc.', 'Philips Digital Mammography Sweden AB', 'GE MEDICAL SYSTEMS', 'Philips Medical Systems', 'SIEMENS']
    # view = image.view # MLO_VIEW = ['MLO','LMLO','RMLO', 'LMO', 'ML'] CC_VIEW = ['CC','LCC','RCC', 'XCCL', 'XCCM']
    laterality = image.laterality # L R
    if ".dcm" in image.path:
        # If this doesn't work. Check Lidia's https://gitlab.com/eucanimage/BreastCancer/-/blob/master/src/preprocessing/mmg_utils.py
        img_dcm = dicom.dcmread(image.path)
        img_np = img_dcm.pixel_array
    else:
        img_pil = Image.open(image.path).convert('RGB')
        img_np = np.array(img_pil)
    scale_size = (rescale_height, rescale_width)
    if len(img_np.shape) > 2:
        img_np = img_np.transpose(2,0,1)[0] # remove redundant channel dimensions
        # return img_np # Uncommment to omit preprocessing (for statistics purposes...)
    # CROP!
    img_np = crop_MG(img_np) # Works for CMMD.
    img_np = np.uint8(img_np) if img_np.dtype != np.uint8 else img_np.copy()
    rescaled_img, scale_factor = imrescale(img_np, scale_size, return_scale=True, backend='pillow')
    # rescaled_img = img_np # return image # CMMD dimensionality statistics required this

    # if rescaled_img[50][0] == 0: # To eliminate laterality bias
    if laterality == 'R' or laterality == 'RIGHT':
        rescaled_img = np.fliplr(rescaled_img)

    image = torch.from_numpy(rescaled_img.copy()).unsqueeze(0)
    image = image.repeat(3,1,1) # Convert to 3 channels just because for some reason there's no better solution yet. https://github.com/pytorch/vision/issues/1732
    
    # return image # CMMD dimensionality statistics required this
    # Histogram Matching 
    landmarks_values = torch.load(HOME_PATH / LANDMARKS)
    apply_hist_stand_landmarks(image, landmarks_values)

    # Images need to be same size. So pad with zeros after cropping. Maybe rescaling is better? Not sure.
    paddedimg = torch.zeros(3,224,224) # There are inconsistencies between datasets. So we negate the crop. Yeah we came full circle. What can you do.
    c,h,w = image.shape
    # if image[0][50][0] == 0:
    #     paddedimg[:,-h:,-w:] = image
    # else:
    #     paddedimg[:,:h,:w] = image
    paddedimg[:,:h,:w] = image

    # paddedimg[:,-h:,-w:]=image

    return paddedimg, label

class ALLDataset(): # Should work for any center
    def __init__(self, dataset_path, csv_path, data_loader_type='optimam', mode='train', load_max=-1, batch_size=10, center=None): 
        if data_loader_type == 'optimam':
            subjects = OPTIMAMDataset(csv_path, dataset_path, detection=False, load_max=load_max, 
                                cropped_to_breast=True) # we should be able to load any dataset with this
        elif data_loader_type == 'bcdr':
            # root path is '/home/lidia-garrucho/datasets/BCDR/cropped/ in both cases
            csv_path = [os.path.join(csv_path,'cropped/BCDR-D01_dataset/dataset_info.csv'),
                        os.path.join(csv_path,'cropped/BCDR-D02_dataset/dataset_info.csv'),
                        os.path.join(csv_path,'cropped/BCDR-DN01_dataset/dataset_info.csv')]
            dataset_path = [os.path.join(dataset_path,'cropped/BCDR-D01_dataset'),
                            os.path.join(dataset_path,'cropped/BCDR-D02_dataset'),
                            os.path.join(dataset_path,'cropped/BCDR-DN01_dataset')]
            subjects = BCDRDataset(csv_path, dataset_path, detection=False, load_max=load_max, 
                                cropped_to_breast=True)
        elif data_loader_type == 'inbreast':
            # csv_path = '/home/lidia-garrucho/datasets/INBREAST/INbreast_updated_cropped_breast.csv'
            # dataset_path = '/home/lidia-garrucho/datasets/INBREAST/AllPNG_cropped'
            subjects = INBreastDataset(csv_path, dataset_path, detection=False, load_max=load_max, 
                                cropped_to_breast=True)
        elif data_loader_type == 'general' or 'cmmd':
            subjects = CMMDDataset(csv_path, dataset_path, load_max=load_max)
        
        # subjects_selected = {}
        if center!=None:
            # In case the dataset is multi-centric
            total_subjects = subjects.get_images_by_site(center)
        # else:
        #     # General case
        #     subjects_selected['benign'] = subjects.get_clients_by_status('Benign')[:load_max] #Note that clients means subjects here.
        #     subjects_selected['malignant'] = subjects.get_clients_by_status('Malignant')[:load_max]
        #     subjects_selected['normal'] = subjects.get_clients_by_status('Normal')[:load_max]
        #     if CONFIG['data']['balance']:
        #         balance_to_min_index = min([len(subjects_selected['benign']), len(subjects_selected['malignant'])]) #, len(subjects_selected['normal'])])
        #         subjects_selected['benign'] = subjects_selected['benign'][:balance_to_min_index]
        #         subjects_selected['malignant'] = subjects_selected['malignant'][:balance_to_min_index]
        #         # subjects_selected['normal'] = subjects_selected['normal'][:balance_to_min_index]
        #         total_subjects = subjects_selected['benign'] + subjects_selected['malignant'] # + subjects_selected['normal']
        #         for status in ['benign', 'malignant']:
        #             print(f'Total subjects selected by status ({status}): {len(subjects_selected[status])}')
        #     else:
        #         total_subjects = subjects_selected['benign'] + subjects_selected['malignant'] + subjects_selected['normal']
        #         for status in ['normal', 'benign', 'malignant']:
        #             print(f'Total subjects selected by status ({status}): {len(subjects_selected[status])}')
        else:
            images_benign, images_normal, images_malignant = [], [], []
            for c in subjects:
                for imlist, status in zip([images_normal, images_benign, images_malignant], ['Normal', 'Benign', 'Malignant']):
                    client_images_by_status = c.get_images_by_status(status=[status])
                    for image in client_images_by_status:
                        imlist.append(image)
            if CONFIG['data']['balance']:
                # Balance the dataset
                balance_to_min_index = min([len(images_benign), len(images_malignant)])
                images_benign = images_benign[:balance_to_min_index]
                images_malignant = images_malignant[:balance_to_min_index]
                total_images = images_benign + images_malignant
                for ims, status in zip([images_benign, images_malignant], ['benign', 'malignant']):
                    print(f'Total images selected by status ({status}): {len(ims)}')
            else:
                total_images = images_benign + images_malignant + images_normal
                for ims, status in zip([images_benign, images_malignant], ['benign', 'malignant']):
                    print(f'Total images selected by status ({status}): {len(ims)}')
                        
        
        random.shuffle(total_images) 
        
        # Data Split
        training_images = total_images[:int(0.8*len(total_images))]
        validation_images = total_images[int(0.8*len(total_images)):]
        # test_images = total_images[int(0.8*len(total_images)):]

        if mode == 'train':
            self.images = training_images
        elif mode == 'validation' or 'val':
            self.images = validation_images
        elif mode == 'test':
            self.images = test_images
        else:
            raise ValueError(f'Mode: "{mode}" not recognized')

        # training_subjects = total_subjects[:int(len(total_subjects)*0.8)]
        # validation_subjects = total_subjects[int(len(total_subjects)*0.8):int(len(total_subjects)*0.9)]
        # test_subjects = total_subjects[int(len(total_subjects)*0.9):]

        # def extract_images(subjects):
        #     images=[]
        #     for subject in tqdm(subjects):
        #         for study in subject:
        #             for image in study:
        #                 images.append(image)
        #     return images

        # if mode == 'train':
        #     self.images = extract_images(training_subjects)
        # elif mode == 'val':
        #     self.images = extract_images(validation_subjects) 
        # elif mode == 'test':
        #     self.images = extract_images(test_subjects)

        assert len(self.images)>0, "No images found in the dataset. Something is wrong with the input path or your dataloader choice."

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