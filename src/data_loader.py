seed = 42  # for reproducibility

# Imports
import os
import yaml
import enum
import copy
import random
random.seed(seed)
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

# To log the selected centers. Uncomment if you need to reinitialize the dicts to store the ids of the selected center images.
# centralized_image_ids_dict = {'jarv':{'train':[], 'val':[]}, 'stge':{'train':[], 'val':[]}, 'inbreast':{'train':[], 'val':[]}, 'bcdr':{'train':[], 'val':[]}, 'cmmd':{'train':[], 'val':[]}}

if not os.path.exists("image_ids.pkl") and CONFIG['simulation'] == False:
    image_ids_dict = {'jarv':{'train':[], 'val':[]}, 'stge':{'train':[], 'val':[]}, 'inbreast':{'train':[], 'val':[]}, 'bcdr':{'train':[], 'val':[]}, 'cmmd':{'train':[], 'val':[]}}
    with open("image_ids.pkl", 'wb') as handle: # To make sure they are the same
        pickle.dump(image_ids_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
# if not os.path.exists("federated_image_ids.pkl"):
#     with open("federated_image_ids.pkl", 'wb') as handle:
#         pickle.dump(federated_image_ids_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open("federated_image_ids.pkl", 'rb') as handle:
#     federated_image_ids_dict = pickle.load(handle)

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
        raise ValueError("Unknown status: {}".format(image.status))

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
    img_np = crop_MG(img_np) # This is for CMMD. Doesn't harm if the image is already cropped.
    img_np = np.uint8(img_np) if img_np.dtype != np.uint8 else img_np.copy()
    rescaled_img, scale_factor = imrescale(img_np, scale_size, return_scale=True, backend='pillow')
    # rescaled_img = img_np # return image # CMMD dimensionality statistics required this

    # if rescaled_img[50][0] == 0: # To eliminate laterality bias
    if CONFIG['data']['flip']:
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

    # Random Flip
    if CONFIG['data']['flip']:
        if random.random() > 0.5:
            paddedimg = paddedimg.flip(2)
    
    if CONFIG['data']['rotate']:
        if random.random() > 0.5:
            if random.random() > 0.5:
                paddedimg = np.rot90(paddedimg,k=1)
            else:
                paddedimg = np.rot90(paddedimg,k=3)

    return paddedimg, label

class ALLDataset(): # Should work for any center
    def __init__(self, dataset_path, csv_path, data_loader_type='stge', mode='train', load_max=-1, batch_size=10): 
        
        # OPTIMAM(jarv)
        jarv_csv_path="/home/akis-linardos/Datasets/OPTIMAM/jarv_info.csv"
        jarv_dataset_path="/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/images"

        # OPTIMAM(stge)
        stge_csv_path="/home/akis-linardos/Datasets/OPTIMAM/stge_info.csv"
        stge_dataset_path="/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/images"

        # BCDR
        # root path is '/home/lidia-garrucho/datasets/BCDR/cropped/ in both cases
        bcdr_csv_path = [os.path.join('/home/lidia-garrucho/datasets/BCDR','cropped/BCDR-D01_dataset/dataset_info.csv'),
                    os.path.join('/home/lidia-garrucho/datasets/BCDR','cropped/BCDR-D02_dataset/dataset_info.csv'),
                    os.path.join('/home/lidia-garrucho/datasets/BCDR','cropped/BCDR-DN01_dataset/dataset_info.csv')]
        bcdr_dataset_path = [os.path.join("/home/lidia-garrucho/datasets/BCDR",'cropped/BCDR-D01_dataset'),
                        os.path.join("/home/lidia-garrucho/datasets/BCDR",'cropped/BCDR-D02_dataset'),
                        os.path.join("/home/lidia-garrucho/datasets/BCDR",'cropped/BCDR-DN01_dataset')]

        # InBreast
        inbreast_csv_path = '/home/lidia-garrucho/datasets/INBREAST/INbreast_updated_cropped_breast.csv'
        inbreast_dataset_path = '/home/lidia-garrucho/datasets/INBREAST/AllPNG_cropped'

        # CMMD
        cmmd_csv_path='/home/akis-linardos/Datasets/CMMD/info.csv'
        cmmd_dataset_path='/home/akis-linardos/Datasets/CMMD'

        if data_loader_type =='jarv':
            subjects_jarv = OPTIMAMDataset(jarv_csv_path, jarv_dataset_path, detection=False, load_max=load_max, 
                cropped_to_breast=True) # we should be able to load any dataset with this
            subjects = subjects_jarv

        elif data_loader_type == 'stge':
            subjects_stge = OPTIMAMDataset(stge_csv_path, stge_dataset_path, detection=False, load_max=load_max,
                cropped_to_breast=True)
            subjects = subjects_stge
        elif data_loader_type == 'bcdr':
            subjects_bcdr = BCDRDataset(bcdr_csv_path, bcdr_dataset_path, detection=False, load_max=load_max, 
                cropped_to_breast=True)
            subjects = subjects_bcdr
        elif data_loader_type == 'inbreast':
            subjects_inbreast = INBreastDataset(inbreast_csv_path, inbreast_dataset_path, detection=False, load_max=load_max, 
                cropped_to_breast=True)
            subjects = subjects_inbreast
        elif data_loader_type == 'cmmd':
            subjects_cmmd = CMMDDataset(cmmd_csv_path, cmmd_dataset_path, load_max=load_max)
            subjects = subjects_cmmd
        elif data_loader_type == 'general': # for unseen centers
            subjects_general = CMMDDataset(csv_path, dataset_path, load_max=load_max)
            subjects = subjects_general
        elif data_loader_type == 'all': # For Centralized experiments
            subjects_jarv = OPTIMAMDataset(jarv_csv_path, jarv_dataset_path, detection=False, load_max=load_max,
                cropped_to_breast=True)
            subjects_stge = OPTIMAMDataset(stge_csv_path, stge_dataset_path, detection=False, load_max=load_max,
                cropped_to_breast=True)
            subjects_bcdr = BCDRDataset(bcdr_csv_path, bcdr_dataset_path, detection=False, load_max=load_max,
                cropped_to_breast=True)
            subjects_inbreast = INBreastDataset(inbreast_csv_path, inbreast_dataset_path, detection=False, load_max=load_max,
                cropped_to_breast=True)
            subjects_cmmd = CMMDDataset(cmmd_csv_path, cmmd_dataset_path, load_max=load_max)
            subjects = [subjects_jarv, subjects_stge, subjects_bcdr, subjects_inbreast, subjects_cmmd] # to balance each
            subjects_center = ['jarv', 'stge', 'bcdr', 'inbreast', 'cmmd']
            # subjects = subjects_stge + subjects_jarv + subjects_bcdr + subjects_inbreast + subjects_cmmd
        print("csv_path is", csv_path)
        print("dataset_path is", dataset_path)
        print("subjects length is", len(subjects))

        # In the simulation we use the extracted IDs. Real world should use the other function still
        def get_images_from_subjects_simulation(subjects_f, image_id_list):
            images_benign, images_normal, images_malignant = [], [], []
            for c in subjects_f:
                for imlist, status in zip([images_normal, images_benign, images_malignant], ['Normal', 'Benign', 'Malignant']):
                    client_images_by_status = c.get_images_by_status(status=[status])
                    for image in client_images_by_status:
                        imlist.append(image)
            if CONFIG['data']['balance']:
                # Balance the dataset
                balance_to_min_index = min([len(images_benign), len(images_malignant)])
                if not CONFIG['data']['max_per_label']:
                    images_benign = images_benign[:balance_to_min_index]
                    images_malignant = images_malignant[:balance_to_min_index]
                else:
                    images_benign = images_benign[:CONFIG['data']['max_per_label']]
                    images_malignant = images_malignant[:CONFIG['data']['max_per_label']]
                total_images = images_benign + images_malignant
                for ims, status in zip([images_benign, images_malignant], ['benign', 'malignant']):
                    print(f'Total images selected by status ({status}): {len(ims)}')
            else:
                total_images = images_benign + images_malignant + images_normal
                for ims, status in zip([images_benign, images_malignant], ['benign', 'malignant']):
                    print(f'Total images selected by status ({status}): {len(ims)}')
            # random.shuffle(total_images) 
            # Data Split
            images_to_use = []
            # if mode == 'train':
            #     for image in total_images:
            #         if image.id not in validation_image_id_list:
            #             images_to_use.append(image)
            # elif mode == 'validation' or 'val':
            #     for image in total_images:
            #         if image.id in validation_image_id_list:
            #             images_to_use.append(image)

            for image in total_images:
                if image.id in image_id_list:
                    images_to_use.append(image)
            img_ids = (mode, [image.id for image in images_to_use])
            random.shuffle(images_to_use)

            return images_to_use
        
        def get_images_from_subjects(subjects_f):
            images_benign, images_normal, images_malignant = [], [], []
            for c in subjects_f:
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
                images_to_use = training_images
            elif mode == 'validation' or 'val':
                images_to_use = validation_images
            elif mode == 'test':
                images_to_use = test_images
            else:
                raise ValueError(f'Mode: "{mode}" not recognized')

            img_ids = [image.id for image in images_to_use]

            return images_to_use, img_ids
              
        self.data_loader_type = data_loader_type

        if CONFIG['simulation']==True:

            if data_loader_type == 'all': # CDS
                
                with open('image_ids.pkl', 'rb') as handle:
                    image_ids_dict = pickle.load(handle)

                self.images = []
                for i, s in enumerate(subjects):
                    # images_to_use = get_images_from_subjects(s)[0] #_simulation(s, image_ids_dict[subjects_center[i]]['val'])

                    if mode == 'train':
                        images_to_use = get_images_from_subjects_simulation(s, image_ids_dict[subjects_center[i]]['train'])
                    elif mode == 'validation' or 'val':
                        images_to_use = get_images_from_subjects_simulation(s, image_ids_dict[subjects_center[i]]['val'])
                    
                    images_with_center = [(img,subjects_center[i]) for img in images_to_use]
                    self.images = self.images+images_with_center
                    # centralized_image_ids_dict[subjects_center[i]][mode] = img_ids
                    
            else:
                
                with open('image_ids.pkl', 'rb') as handle:
                    image_ids_dict = pickle.load(handle)

                if mode == 'train':
                    self.images = get_images_from_subjects_simulation(subjects, image_ids_dict[data_loader_type]['train'])
                elif mode == 'validation' or 'val':
                    self.images = get_images_from_subjects_simulation(subjects, image_ids_dict[data_loader_type]['val'])

                # self.images = get_images_from_subjects(subjects)[0] #_simulation(subjects, image_ids_dict[data_loader_type]['val'])
                # federated_image_ids_dict[data_loader_type][mode] = img_ids
        
        else:
            # with open("image_ids.pkl", 'rb') as handle:
            #     image_ids_dict = pickle.load(handle)
            self.images, image_ids = get_images_from_subjects(subjects)
            
            # image_ids_dict[data_loader_type][mode] = image_ids
            # with open("image_ids.pkl", 'wb') as handle:
            #     pickle.dump(image_ids_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # with open('federated_image_ids.pkl', 'wb') as handle:
            #     pickle.dump(federated_image_ids_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
        
        if self.data_loader_type == 'all':
            if isinstance(idx, slice):
                # do your handling for a slice object:
                return [(preprocess_one_image_OPTIMAM(image), center) for image, center in self.images[idx]]
            else:
                # Do your handling for a plain index
                image, center = self.images[idx]
                return (preprocess_one_image_OPTIMAM(image), center)

        else:
            if isinstance(idx, slice):
                # do your handling for a slice object:
                return [preprocess_one_image_OPTIMAM(image) for image in self.images[idx]]
            else:
                # Do your handling for a plain index
                image = self.images[idx]
                return preprocess_one_image_OPTIMAM(image)

def test_center_data(dataset_path, csv_path, mode, load_max=1):
    subjects = CMMDDataset(csv_path, dataset_path, load_max=load_max)
    print("csv_path is", csv_path)
    print("dataset_path is", dataset_path)
    print("subjects length is", len(subjects))
    def get_images_from_subjects(subjects_f):
        images_benign, images_normal, images_malignant = [], [], []
        for c in subjects_f:
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
            images_to_use = training_images
        elif mode == 'validation' or 'val':
            images_to_use = validation_images
        elif mode == 'test':
            images_to_use = test_images
        else:
            raise ValueError(f'Mode: "{mode}" not recognized')

        img_ids = [image.id for image in images_to_use]

        return images_to_use, img_ids

    images, image_ids = get_images_from_subjects(subjects)
    print(f"Number of images selected for mode {mode} is {len(images)}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset_path", type=str, help="The full path to your dataset")
    parser.add_argument("-c","--csv_path", type=str, help="The full path to your csv file")
    args = parser.parse_args()
    test_center_data(args.dataset_path, args.csv_path, mode='train')

    