import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch
from src.data_handling.optimam_dataset import OPTIMAMDataset
from src.visualizations.plot_image import plot_image_opencv_fit_window
from src.data_augmentation.breast_density.transforms import *
from src.data_augmentation.breast_density.models import create_model
from src.data_augmentation.breast_density.data.base_dataset import get_params, get_transform
from src.data_augmentation.breast_density.data.resize_image import *

def init_cycleGAN(image_height, checkpoints_dir, checkpoint_name, gpu_ids):

    opt = TestOptions(image_height, checkpoints_dir, checkpoint_name, gpu_ids)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()
    return opt, model


def get_high_density_mammogram(opt, model, image_path, scale_size):

    #print(f'----> img shape {img.shape}') #(1321, 800, 3)
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
    # img_pil = Image.fromarray(img)
    #print(f'----> img_pil size {img_pil.size}')
    input_image = transforms(Image.fromarray(img_rescaled)) # tensor (800, 1320)
    gen = model.netG_A(input_image.unsqueeze(0).to(f'cuda:{opt.gpu_ids[0]}'))
    img_high_density = tensor2np(gen, imtype=np.uint8)
    #print(f'----> out size {out.shape}') #(1320, 800)
    img_high_density_pil = Image.fromarray(img_high_density)
    # img_pil.save(os.path.join('/home/lidia-garrucho/source/mmdetection/data_augmentation/ori_test_cyclegan_1.png'))
    # save_image.save(os.path.join('/home/lidia-garrucho/source/mmdetection/data_augmentation/test_cyclegan_1.png'))
    #np_in = np.array(img_pil)
    # Width and height scales to adjust the BBox coordinates
    w_scale = img_width_rescaled / in_width
    h_scale = img_height_rescaled / in_height

    return img_high_density_pil, h_scale, w_scale


if __name__ == '__main__':

    out_folder = '/home/lidia/Datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density'
    if not Path(out_folder).exists():
        Path(out_folder).mkdir(parents=True)

    checkpoints_dir = '/home/lidia/source/BreastCancer/src/data_augmentation/breast_density/checkpoints'
    checkpoint_name = 'high_density_h800'
    gpu_ids = 0
    rescale_height = 1332
    rescale_width = 800
    opt, model = init_cycleGAN(rescale_height, checkpoints_dir, checkpoint_name, gpu_ids)

    # Cropped scans
    info_csv='/home/lidia/Datasets/OPTIMAM/png_screening_cropped_fixed/client_images_screening.csv'
    dataset_path='/home/lidia/Datasets/OPTIMAM/png_screening_cropped_fixed/images'
    output_path = '/home/lidia/Datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn'
    cropped_scans = True
    fit_to_breast = True

    # Cropped scans GPU Server
    # info_csv='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/client_images_screening.csv'
    # dataset_path='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/images'
    # output_path = '/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn'
    # cropped_scans = True
    # fit_to_breast = True

    detection = True
    load_max = -1
    class_name = 'mass'
    pathologies = ['mass'] #['mass', 'calcifications', 'suspicious_calcifications', 'architectural_distortion'] #['mass'] # None # ['mass']
    use_status = False
    category_id_dict = {'mass': 0}
    
    optimam_clients = OPTIMAMDataset(info_csv, dataset_path, detection=True, load_max=-1, 
                                cropped_scans=cropped_scans)
    clients_selected = optimam_clients.get_clients_by_pathology_and_status(pathologies)
    print(f'Total clients in loaded dataset: {len(optimam_clients)}')
    print(f'Pahologies selected: {pathologies}')
    print('-----------------------------------')
    print(f'Clients: {optimam_clients.total_clients(pathologies)} - Images: {optimam_clients.total_images(pathologies)} - Annotations: {optimam_clients.total_annotations(pathologies)}')
        
    manufacturer = 'HOLOGIC, Inc.'
    image_info_high_density_aug = pd.DataFrame(columns=['image_id', 'h_scale', 'w_scale'])
    image_ctr = 0
    for client in clients_selected:
        images = client.get_images_by_pathology(pathologies)
        for image in images:
            img_high_density_pil, h_scale, w_scale = get_high_density_mammogram(
                                            opt, model, image.path, (rescale_height, rescale_width))
            img_high_density_pil.save(os.path.join(out_folder, image.id +'.png'))
            image_info_high_density_aug.loc[image_ctr] = [image.id] + [h_scale] + [w_scale]
            image_ctr += 1
            if False:
                for annotation in image.annotations:
                    if pathologies is None:
                        add_annotation = True
                    elif any(item in annotation.pathologies for item in pathologies):
                        add_annotation = True
                    else:
                        add_annotation = False
                    if add_annotation:
                        xmin, xmax, ymin, ymax = annotation.get_xmin_xmax_ymin_ymax(fit_to_breast)
                        xmin, xmax = int(xmin * w_scale), int(xmax * w_scale)
                        ymin, ymax = int(ymin * h_scale), int(ymax * h_scale)
                        #poly = [xmin, ymin],[xmax, ymin],[xmax, ymax],[xmin, ymax]
                        poly = [xmin, ymin], [int((xmax-xmin)/2) + xmin, ymin], \
                        [xmax, ymin], [xmax, int((ymax-ymin)/2) + ymin], \
                        [xmax, ymax], [int((xmax-xmin)/2) + xmin, ymax], \
                        [xmin, ymax], [xmin, int((ymax-ymin)/2) + ymin]
                        img = cv2.cvtColor(np.array(img_high_density_pil), cv2.COLOR_BGR2RGB)
                        cv2.polylines(img, [np.array(poly)], True, (0, 255, 0), 2)
                        plt.figure()
                        plt.imshow(img)
                        #plt.imsave('./test.png', img)
                        plt.show()
    
    image_info_high_density_aug.to_csv(os.path.join(out_folder, 'image_info_high_density_aug.csv'), index=False) #, float_format='%.4f')