from pathlib import Path
from typing import Union
import yaml
import numpy as np
import cv2
import torch
import os
import requests
from urllib.request import Request, urlopen
import json
import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict
import logging

## BEGIN CycleGAN code
from PIL import Image
from .src.models import create_model
from .src.data.base_dataset import get_params, get_transform
from .src.data.resize_image import *

class TestOptions:
    def __init__(self, image_height, checkpoints_dir, checkpoint_name, gpu_ids):
        self.gpu_ids = [gpu_ids]
        self.direction = 'AtoB' #'BtoA' high to low
        self.dataroot = None #'/home/lidia-garrucho/datasets/BCDR/style_transfer/1333_800/testA'
        self.preprocess = 'none'
        self.load_size = image_height #1333
        self.crop_size = image_height #1333
        self.input_nc = 1
        self.output_nc = 1
        self.name = checkpoint_name
        self.checkpoints_dir = checkpoints_dir #'/home/lidia-garrucho/source/mmdetection/data_augmentation/checkpoints'
        self.no_dropout = True
        self.model = 'cycle_gan'
        self.num_threads = 0   # test code only supports num_threads = 0
        self.batch_size = 1    # test code only supports batch_size = 1
        self.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        self.no_flip = True    # no flip; comment this line if results on flipped images are needed.
        self.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
        self.isTrain = False
        self.dataset_mode  = 'single'
        self.epoch = 'latest'
        self.eval = False
        self.init_gain = 0.02
        self.init_type = 'normal'
        self.load_iter = 0
        self.max_dataset_size = float("inf")
        self.model_suffix = ''
        self.n_layers_D = 3
        self.ndf = 64
        self.netD = 'basic'
        self.netG = 'resnet_9blocks'
        self.ngf = 64
        self.norm = 'instance'
        self.num_test = 50
        self.phase = 'test'
        self.results_dir = './results/'
        self.suffix = ''
        self.verbose = False

def init_cycleGAN(image_height, checkpoints_dir, checkpoint_name, gpu_ids):
    opt = TestOptions(image_height, checkpoints_dir, checkpoint_name, gpu_ids)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()
    return opt, model

def tensor2np(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        # if image_numpy.shape[0] == 1:  # grayscale to RGB
        #     image_numpy = np.tile(image_numpy, (3, 1, 1))
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        if image_numpy.shape[0] == 1:
            image_numpy = image_numpy.reshape(image_numpy.shape[1], image_numpy.shape[2])
            image_numpy = (image_numpy + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        else:
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def get_high_density_mammogram(opt, model, image_path, scale_size):
    try:
        img_pil = Image.open(image_path).convert('RGB')
        img_np = np.array(img_pil)
        in_height, in_width = img_np.shape[0], img_np.shape[1]
        img_np = np.uint8(img_np) if img_np.dtype != np.uint8 else img_np.copy()
        img, scale_factor = imrescale(img_np, scale_size, return_scale=True, backend='pillow')
        # the w_scale and h_scale has minor difference
        # a real fix should be done in the mmcv.imrescale in the future
        new_h, new_w = img.shape[:2]
        img_rescaled = None
        # Check if multiple of 4 and adjust
        if new_h % 4 or new_w % 4:
            crop_h = new_h % 4
            crop_w = new_w % 4
            img_height_rescaled = new_h - crop_h
            img_width_rescaled = new_w - crop_w
            img_rescaled = np.zeros((img_height_rescaled, img_width_rescaled , img.shape[2]), dtype='uint8')
            img_rescaled = img[0:img_height_rescaled, 0:img_width_rescaled]
        else:
            img_height_rescaled = new_h
            img_width_rescaled = new_w
            img_rescaled = img

        transform_params = get_params(opt, (int(img_width_rescaled / 2), img_height_rescaled))
        transforms = get_transform(opt, transform_params, grayscale=(opt.output_nc == 1))

        input_image = transforms(Image.fromarray(img_rescaled)) # tensor (height, width)
        if opt.gpu_ids[0]:
            gen = model.netG_A(input_image.unsqueeze(0).to(f'cuda:{opt.gpu_ids[0]}'))
        else:
            gen = model.netG_A(input_image.unsqueeze(0))
        img_high_density_np = tensor2np(gen, imtype=np.uint8)
        # Width and height scales to adjust the BBox coordinates
        w_scale = img_width_rescaled / in_width
        h_scale = img_height_rescaled / in_height
    except Exception as e:
        logging.error(f"Error while trying to process {image_path}: {e}")
        raise e

    return img_high_density_np, h_scale, w_scale

## END CycleGAN code

def get_the_model_url(model_name,url):
    model_path = Path(f"my_models/{model_name}.pth")
    if not os.path.exists("my_models/"):
        os.makedirs("my_models/")
    try:
        models_abs_path = model_path.resolve(strict=True)
    except FileNotFoundError:
        try:
            logging.info(f"Downloading {model_name}....")
            r = requests.get(url, allow_redirects=True)
            open(f'my_models/{model_name}.pth', 'wb').write(r.content)
        except Exception as e:
            logging.error(f"Error while trying to the model {model_name}: {e}")
            raise e
    return model_path         

def load_yaml(path):
    with open(path, "r") as yaml_file:
        try:
            return yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            logging.error(exc)
            return {}

def save_generated_images(image_np, output_filepath, input_filepath):
    try:
        Path(os.path.dirname(output_filepath)).mkdir(parents=True, exist_ok=True)
        if os.path.isdir(output_filepath):
            dir_path, input_filename = os.path.split(input_filepath)
            output_filepath = os.path.join(output_filepath, input_filename)
        img_high_density_pil = Image.fromarray(image_np)
        img_high_density_pil.save(output_filepath)
    except Exception as e:
        logging.error(f"Error while trying to output image from {input_filepath}: {e}")
        raise e
    logging.info(f"Saved generated image to {output_filepath}")


def generate_GAN_images(model_file, image_size, num_samples, output_path, gpu_id,
                        input_path, save_images):
    if model_file:
        checkpoints_dir, checkpoint_name = os.path.split(model_file)
        rescale_height = image_size[0] #1332
        rescale_width = image_size[1] #800
        image_size = (rescale_height, rescale_width)
        if gpu_id != -1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if str(device) == 'cuda':
                gpu_ids = gpu_id
            else:
                gpu_ids = None # 'cpu'
        else:
            gpu_ids = None # 'cpu'
        
        # instantiate the model
        logging.info("Instantiating model...")

        rescale_height = image_size[0]
        rescale_width = image_size[1]
        opt, model = init_cycleGAN(rescale_height, checkpoints_dir, checkpoint_name, gpu_ids)

        logging.info("Generating images...")
        # Rescaling of the image inside the method
        if os.path.isdir(input_path):
            output_images = []
            for file in os.listdir(input_path):
                if file.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    input_image = os.path.join(input_path, file)
                    output_image, h_scale, w_scale = get_high_density_mammogram(
                        opt, model, input_image, scale_size=(rescale_height, rescale_width))
                    if save_images:
                        save_generated_images(output_image, output_path, input_image)
                    else:
                        output_images.append(output_image)
            if not save_images:
                return output_images
        else:
            output_image, h_scale, w_scale = get_high_density_mammogram(
                        opt, model, input_path, scale_size=(rescale_height, rescale_width))
            if save_images:
                save_generated_images(output_image, output_path, input_path)
            else:
                return output_image
    else:
        logging.info("we can't find this type of GAN")
