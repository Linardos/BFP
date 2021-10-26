import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
# from data_augmentation.options.test_options import TestOptions
from src.data_augmentation.breast_density.models import create_model
from src.data_augmentation.breast_density.data.base_dataset import get_params, get_transform
import torch 
import cv2, glob
import random

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


class TestOptions:
    def __init__(self, image_height, checkpoints_dir, checkpoint_name, gpu_ids):
        self.gpu_ids = [gpu_ids]
        self.direction = 'AtoB'
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


class Low2HighBreastDensityAug:

    def __init__(self, checkpoint_name, img_scale, to_rgb=False):
        self.checkpoint_name = checkpoint_name
        self.img_scale = img_scale
        self.to_rgb = to_rgb
        self.out_width = img_scale[1]
        self.out_height = img_scale[0]
        self.apply = True
        self.transforms = None
        self.opt = TestOptions(checkpoint_name, 0)
        self.model = create_model(self.opt)      # create a model given opt.model and other options
        self.model.setup(self.opt)               # regular setup: load and print networks; create schedulers
        self.model.eval()

    def __call__(self, results):

        apply = np.random.choice([True, False])
        if apply:
            for key in results.get('img_fields', ['img']):
                img_u8 = results[key]
                #print(img_u8.dtype)
                img = np.uint8(img_u8) if img_u8.dtype != np.uint8 else img_u8.copy()
                #print(f'----> img shape {img.shape}') #(1321, 800, 3)
                if self.transforms is None:
                    transform_params = get_params(self.opt, (int(img.shape[1] / 2), img.shape[0]))
                    self.transforms = get_transform(self.opt, transform_params, grayscale=(self.opt.output_nc == 1))
                    #print(self.transforms)
                img_pil = Image.fromarray(img)
                #print(f'----> img_pil size {img_pil.size}')
                input_image = self.transforms(img_pil) # tensor (800, 1320)
                gen = self.model.netG_A(input_image.unsqueeze(0).to(f'cuda:{self.opt.gpu_ids[0]}'))
                out = tensor2np(gen, imtype=np.uint8)
                #print(f'----> out size {out.shape}') #(1320, 800)
                # save_image = Image.fromarray(out)
                # img_pil.save(os.path.join('/home/lidia-garrucho/source/mmdetection/data_augmentation/ori_test_cyclegan_1.png'))
                # save_image.save(os.path.join('/home/lidia-garrucho/source/mmdetection/data_augmentation/test_cyclegan_1.png'))
                #np_in = np.array(img_pil)
                AtoB_img = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype='uint8')
                if img.shape[0] < out.shape[0] or img.shape[1] < out.shape[1]:
                    min_height = min(img.shape[0], out.shape[0])
                    min_width = min(img.shape[1], out.shape[1])
                    AtoB_img[0:min_height, 0:min_width, 0] = out[0:min_height, 0:min_width]
                    if img.shape[2] == 3:
                        AtoB_img[0:min_height, 0:min_width, 1] = out[0:min_height, 0:min_width]
                        AtoB_img[0:min_height, 0:min_width, 2] = out[0:min_height, 0:min_width]
                else:
                    AtoB_img[0:out.shape[0], 0:out.shape[1], 0] = out
                    if img.shape[2] == 3:
                        AtoB_img[0:out.shape[0], 0:out.shape[1], 1] = out
                        AtoB_img[0:out.shape[0], 0:out.shape[1], 2] = out
                # print(f'----> AtoB_img size {AtoB_img.shape}')
                # AtoB_pil = Image.fromarray(AtoB_img)
                # AtoB_pil.save(os.path.join('/home/lidia-garrucho/source/mmdetection/data_augmentation/color_test_cyclegan_1.png'))
                # exit(0)
                #AtoB_img = AtoB_img.astype(np.float32)
                results[key] = AtoB_img
                # print(f'----> IN')
                # self.apply = False
        # else:
            # print(f'----> OUT')
        #     self.apply = True
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(checkpoint_name={self.checkpoint_name})'
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'to_rgb={self.to_rgb}, '
        return repr_str