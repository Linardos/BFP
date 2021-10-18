import os
import numpy as np
from PIL import Image
from src.data_handling.optimam_dataset import OPTIMAMDataset
from src.visualizations.plot_image import plot_image_opencv_fit_window
from src.data_augmentation.breast_density.data.resize_image import *

if __name__ == '__main__':

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
    plot_images = True

    optimam_clients = OPTIMAMDataset(info_csv, dataset_path, detection=True, load_max=-1, 
                                cropped_scans=cropped_scans)
    clients_selected = optimam_clients.get_clients_by_pathology_and_status(pathologies)
    print(f'Total clients in loaded dataset: {len(optimam_clients)}')
    print(f'Pahologies selected: {pathologies}')
    print('-----------------------------------')
    print(f'Clients: {optimam_clients.total_clients(pathologies)} - Images: {optimam_clients.total_images(pathologies)} - Annotations: {optimam_clients.total_annotations(pathologies)}')
        
    manufacturer = 'HOLOGIC, Inc.'

    image_ctr = 0
    for client in clients_selected:
        images = client.get_images_by_pathology(pathologies)
        for image in images:
            status = image.status # ['Benign', 'Malignant', 'Interval Cancer', 'Normal']
            site = image.site # ['adde', 'jarv', 'stge']
            manufacturer = image.manufacturer # ['HOLOGIC, Inc.', 'Philips Digital Mammography Sweden AB', 'GE MEDICAL SYSTEMS', 'Philips Medical Systems', 'SIEMENS']
            view = image.view # MLO_VIEW = ['MLO','LMLO','RMLO', 'LMO', 'ML'] CC_VIEW = ['CC','LCC','RCC', 'XCCL', 'XCCM']
            laterality = image.laterality # L R

            img_pil = Image.open(image.path).convert('RGB')
            img_np = np.array(img_pil)
            scale_size = (rescale_height, rescale_width)
            img_np = np.uint8(img_np) if img_np.dtype != np.uint8 else img_np.copy()
            rescaled_img, scale_factor = imrescale(img_np, scale_size, return_scale=True, backend='pillow')

            if plot_images:
                plot_image_opencv_fit_window(rescaled_img, title='Input Image', 
                                                screen_resolution=(1920, 1080), wait_key=True)