import os
import numpy as np
from PIL import Image
from src.data_handling.mmg_detection_datasets import *
from src.visualizations.plot_image import plot_image_opencv_fit_window
from src.data_augmentation.breast_density.data.resize_image import *
import torch

from src.preprocessing.histogram_standardization import get_hist_stand_landmarks, apply_hist_stand_landmarks

if __name__ == '__main__':

    # Cropped scans GPU Server
    info_csv='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/client_images_screening.csv'
    dataset_path='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/images'

    info_csv='/home/lidia/Datasets/OPTIMAM/png_screening_cropped_fixed/client_images_screening.csv'
    dataset_path='/home/lidia/Datasets/OPTIMAM/png_screening_cropped_fixed/images'
    cropped_to_breast = True
    
    detection = False
    load_max = 100 #10 Only loads 10 images
    pathologies = None #['mass', 'calcifications', 'suspicious_calcifications', 'architectural_distortion'] # None to select all
    status = ['Normal', 'Benign', 'Malignant'] #['Normal'] 
    # Resize images keeping aspect ratio
    rescale_height = 224 #1333
    rescale_width = 224 #800
    plot_images = True

    # Call to the OPTIMAM Dataloader
    optimam_clients = OPTIMAMDataset(info_csv, dataset_path,detection=detection, load_max=load_max, 
                            cropped_to_breast=cropped_to_breast)
    
    for status in ['Normal', 'Benign', 'Malignant']:
        clients_selected = optimam_clients.get_clients_by_status(status)
        print(f'Total clients selected by status ({status}): {len(clients_selected)}')

    # If you want to select images by center:
    clients_selected = optimam_clients.get_images_by_site('stge')
    print(f'Total clients selected: {len(clients_selected)}')

    # If we don't select clients by pathology, status or site, the loop will be:
    # for client in optimam_clients:
    #     for study in client.studies:
    #         for image in study.images:

    image_ctr = 0
    for client in clients_selected:
        for study in client.studies:
            for image in study:
                status = image.status # ['Benign', 'Malignant', 'Normal']
                site = image.site # ['adde', 'jarv', 'stge']
                manufacturer = image.manufacturer # ['HOLOGIC, Inc.', 'Philips Digital Mammography Sweden AB', 'GE MEDICAL SYSTEMS', 'Philips Medical Systems', 'SIEMENS']
                # view = image.view # MLO_VIEW = ['MLO','LMLO','RMLO', 'LMO', 'ML'] CC_VIEW = ['CC','LCC','RCC', 'XCCL', 'XCCM']
                # laterality = image.laterality # L R

                img_pil = Image.open(image.path).convert('RGB')
                img_np = np.array(img_pil)
                scale_size = (rescale_height, rescale_width)
                img_np = np.uint8(img_np) if img_np.dtype != np.uint8 else img_np.copy()
                rescaled_img, scale_factor = imrescale(img_np, scale_size, return_scale=True, backend='pillow')
                # Pad the image if you need a constant size
                if plot_images:
                    plot_image_opencv_fit_window(rescaled_img, title='Input Image', 
                                                    screen_resolution=(1920, 1080), wait_key=True)
                
    # Get your train split of clients:
    train_clients = [] # TODO
    # Compute landmarks for Histogram Standardization on the train set (subset of 'stge' site):
    image_paths = []
    for client in train_clients:
        selected_images = client.get_images_by_pathology(pathologies)
        for image in selected_images:
            if image.site == 'stge':
                image_paths.append(image.path)

    landmarks_path = 'train_stge_landmarks.pth'
    landmarks = get_hist_stand_landmarks(image_paths)
    torch.save(landmarks, landmarks_path)
    
    landmarks_values = torch.load(landmarks_path)
    # Test Histogram Standardization in another site
    for client in clients_selected:
        images = client.get_images_by_pathology(['mass'])
        for image in images:
            if image.site == 'adde':
                nyul_img = apply_hist_stand_landmarks(Image.open(image.path), landmarks_values)
                nyul_img = nyul_img.astype(np.uint8)
