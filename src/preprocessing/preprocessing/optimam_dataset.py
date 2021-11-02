import os
import sys
import numpy as np
from pathlib import Path
import yaml
import nibabel as nib
import cv2
import pandas as pd
import png
import omidb
import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from src.visualizations.plot_image import plot_image_opencv_fit_window

def process_mark_info(mark):
    pathologies = []
    if mark.architectural_distortion:
        pathologies.append('architectural_distortion')
    if mark.dystrophic_calcification:
        pathologies.append('dystrophic_calcification')
    if mark.fat_necrosis:
        pathologies.append('fat_necrosis')
    if mark.focal_asymmetry:
        pathologies.append('focal_asymmetry')
    if mark.mass:
        pathologies.append('mass')
    if mark.suspicious_calcifications:
        pathologies.append('suspicious_calcifications')
    if mark.milk_of_calcium:
        pathologies.append('milk_of_calcium')
    if mark.other_benign_cluster:
        pathologies.append('other_benign_cluster')
    if mark.plasma_cell_mastitis:
        pathologies.append('plasma_cell_mastitis')
    if mark.benign_skin_feature:
        pathologies.append('benign_skin_feature')
    if mark.calcifications:
        pathologies.append('calcifications')
    if mark.suture_calcification:
        pathologies.append('suture_calcification')
    if mark.vascular_feature:
        pathologies.append('vascular_feature')
    if mark.benign_classification:
        pathologies.append(mark.benign_classification.value)
    if mark.mass_classification:
        pathologies.append(mark.mass_classification.value)
    return pathologies


def convert_optimam_dicom_to_nifti(config, client_list, output_folder, first_run=False):

    dataset_path = config['paths']['dataset']
    if client_list:
        db = omidb.DB(dataset_path, clients=client_list)
    else:
        db = omidb.DB(dataset_path)
    
    # Make output folders
    if not Path(output_folder).exists():
        Path(output_folder).mkdir(parents=True)
    clients = [client for client in db]
    clients_csv = os.path.join(output_folder, 'client_images_screening.csv')
    for client in clients:
        for episode in client.episodes:
            if episode.studies:
                for study in episode.studies:
                    if study.series and study.event_type:
                        if str(study.event_type[0].value) == 'screening':
                            for serie in study.series:
                                if len(serie.images):
                                    for image in serie.images:
                                        if Path(image.dcm_path).exists() and 'PresentationIntentType' in image.dcm:
                                            if image.dcm.PresentationIntentType == 'FOR PRESENTATION':
                                                clients_dict = {'client_id': [], 'status':[], 'site':[], 'study_id':[],
                                                                'serie_id':[], 'image_id':[],
                                                                'view': [], 'laterality': [], 'age': [],
                                                                'mark_id': [], 'lesion_id': [],
                                                                'conspicuity': [], 'x1': [], 'x2': [], 'y1': [],
                                                                'y2': [], 'pathologies': [],
                                                                'manufacturer': [], 'pixel_spacing': [],
                                                                'magnification_factor': [], 'implant': []}
                                                if image.marks:
                                                    for mark in image.marks:
                                                        clients_dict['client_id'].append(client.id)
                                                        clients_dict['status'].append(client.status.value)
                                                        clients_dict['site'].append(client.site.value)
                                                        clients_dict['study_id'].append(study.id)
                                                        clients_dict['serie_id'].append(serie.id)
                                                        clients_dict['image_id'].append(image.id)
                                                        if 'ViewPosition' in image.dcm:
                                                            clients_dict['view'].append(image.dcm.ViewPosition)
                                                        else:
                                                            clients_dict['view'].append('')
                                                        if 'ImageLaterality' in image.dcm:
                                                            clients_dict['laterality'].append(image.dcm.ImageLaterality)
                                                        else:
                                                            clients_dict['laterality'].append('')
                                                        clients_dict['mark_id'].append(mark.id)
                                                        clients_dict['lesion_id'].append(mark.lesion_id)
                                                        if mark.conspicuity:
                                                            clients_dict['conspicuity'].append(mark.conspicuity.value)
                                                        else:
                                                            clients_dict['conspicuity'].append('')
                                                        clients_dict['x1'].append(mark.boundingBox.x1)
                                                        clients_dict['x2'].append(mark.boundingBox.x2)
                                                        clients_dict['y1'].append(mark.boundingBox.y1)
                                                        clients_dict['y2'].append(mark.boundingBox.y2)
                                                        clients_dict['pathologies'].append(" ".join(process_mark_info(mark)))
                                                        if 'BreastImplantPresent' in image.dcm:
                                                            clients_dict['implant'].append(image.dcm.BreastImplantPresent)
                                                        else:
                                                            clients_dict['implant'].append('')
                                                        if 'PatientAge' in image.dcm:
                                                            clients_dict['age'].append(int(image.dcm.PatientAge.replace('Y','')))
                                                        else:
                                                            clients_dict['age'].append('')
                                                        if 'ImagerPixelSpacing' in image.dcm:
                                                            clients_dict['pixel_spacing'].append(f'{image.dcm.ImagerPixelSpacing[0]} {image.dcm.ImagerPixelSpacing[1]}')
                                                        else:
                                                            clients_dict['pixel_spacing'].append('')
                                                        if 'EstimatedRadiographicMagnificationFactor' in image.dcm:
                                                            clients_dict['magnification_factor'].append(image.dcm.EstimatedRadiographicMagnificationFactor)
                                                        else:
                                                            clients_dict['magnification_factor'].append('')
                                                        clients_dict['manufacturer'].append(image.dcm.Manufacturer)
                                                else:
                                                    clients_dict['client_id'].append(client.id)
                                                    clients_dict['status'].append(client.status.value)
                                                    clients_dict['site'].append(client.site.value)
                                                    clients_dict['study_id'].append(study.id)
                                                    clients_dict['serie_id'].append(serie.id)
                                                    clients_dict['image_id'].append(image.id)
                                                    if 'ViewPosition' in image.dcm:
                                                        clients_dict['view'].append(image.dcm.ViewPosition)
                                                    else:
                                                        clients_dict['view'].append('')
                                                    if 'ImageLaterality' in image.dcm:
                                                        clients_dict['laterality'].append(image.dcm.ImageLaterality)
                                                    else:
                                                        clients_dict['laterality'].append('')
                                                    clients_dict['mark_id'].append('')
                                                    clients_dict['lesion_id'].append('')
                                                    clients_dict['conspicuity'].append('')
                                                    clients_dict['x1'].append('')
                                                    clients_dict['x2'].append('')
                                                    clients_dict['y1'].append('')
                                                    clients_dict['y2'].append('')
                                                    clients_dict['pathologies'].append('')
                                                    if 'BreastImplantPresent' in image.dcm:
                                                        clients_dict['implant'].append(image.dcm.BreastImplantPresent)
                                                    else:
                                                        clients_dict['implant'].append('')
                                                    if 'PatientAge' in image.dcm:
                                                        clients_dict['age'].append(int(image.dcm.PatientAge.replace('Y','')))
                                                    else:
                                                        clients_dict['age'].append('')
                                                    if 'ImagerPixelSpacing' in image.dcm:
                                                        clients_dict['pixel_spacing'].append(f'{image.dcm.ImagerPixelSpacing[0]} {image.dcm.ImagerPixelSpacing[1]}')
                                                    else:
                                                        clients_dict['pixel_spacing'].append('')
                                                    if 'EstimatedRadiographicMagnificationFactor' in image.dcm:
                                                        clients_dict['magnification_factor'].append(image.dcm.EstimatedRadiographicMagnificationFactor)
                                                    else:
                                                        clients_dict['magnification_factor'].append('')
                                                    clients_dict['manufacturer'].append(image.dcm.Manufacturer)
                                                    
                                                # Convert DICOM image to Nifti
                                                dcm_path = str(image.dcm_path)
                                                base_path = os.path.dirname(dcm_path.replace(dataset_path, output_folder))
                                                if not Path(base_path).exists():
                                                    Path(base_path).mkdir(parents=True)
                                                scan_filename = os.path.basename(dcm_path.replace(dataset_path, output_folder))[:-4]
                                                nii_image = os.path.join(base_path, scan_filename + '.nii.gz')
                                                if not os.path.exists(nii_image):
                                                    dcim2nii_cmd = "dcm2niix -s y -b n -f " + scan_filename + " -o " + base_path + " -z y " + str(image.dcm_path)
                                                    os.system(dcim2nii_cmd)
                                                else:
                                                    print(f'Skip {scan_filename}')

                                                df = pd.DataFrame.from_dict(clients_dict)
                                                if first_run:
                                                    df.to_csv(clients_csv, index=False)
                                                    first_run = False
                                                else:
                                                    df.to_csv(clients_csv, mode='a', header=False, index=False)                


def fit_to_breast(img_pil):
    low_int_threshold = 0.05
    max_value = 255
    img_8u = (img_pil.astype('float32')/img_pil.max()*max_value).astype('uint8')
    low_th = int(max_value*low_int_threshold)
    _, img_bin = cv2.threshold(img_8u, low_th, maxval=max_value, type=cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_areas = [ cv2.contourArea(cont) for cont in contours ]
    idx = np.argmax(cont_areas)
    x,y,w,h = cv2.boundingRect(contours[idx])
    img_breast = img_pil[y:y+h, x:x+w]
    (xmin, xmax, ymin, ymax) = (x, x + w, y, y + h)
    return img_breast, (xmin, xmax, ymin, ymax)

def insert_cropped_coordinates_to_info_csv(info_csv, cropped_coord_csv):
    info = pd.read_csv(info_csv)
    cropped = pd.read_csv(cropped_coord_csv)
    for index, row in info.iterrows():
        coord = cropped.loc[cropped['image_id'] == row["image_id"]]
        info.loc[index,'xmin_cropped'] = coord['xmin'].values[0]
        info.loc[index,'xmax_cropped'] = coord['xmax'].values[0]
        info.loc[index,'ymin_cropped'] = coord['ymin'].values[0]
        info.loc[index,'ymax_cropped'] = coord['ymax'].values[0]

    info.to_csv(info_csv, index=False)


def convert_optimam_nifti_to_png(info_csv, nifti_folder, png_folder, crop_breast=False):
    
    pathologies = ['mass', 'calcifications', 'architectural_distortion', 'focal_asymmetry',
                    'suspicious_calcifications']
    info = pd.read_csv(info_csv)
    info = info.astype(object).replace(np.nan, '')
    info.head()
    # df = info.groupby(['image_id']).size().reset_index(name='counts')
    unique_image_id_df = info.groupby(['image_id'], as_index=False)

    df_bbox = info.loc[info['mark_id'] != '']
    bbox_counts = df_bbox.groupby(['image_id']).size().reset_index(name='counts')
    print(f'Scans with annotated bboxes: {len(bbox_counts)}')
    pathologies_counter = []
    status_pathologies = []
    for pathology in pathologies:
        count = 0
        status = []
        for i in df_bbox.itertuples():
            if pathology in i.pathologies:
                if i.pathologies.count("suspicious") and i.pathologies.count("calcifications") < 2:
                    if pathology == 'suspicious_calcifications':
                        count += 1
                        status.append(i.status)
                else:
                    count += 1
                    status.append(i.status)
        pathologies_counter.append(count)
        status_pathologies.append(status)

    for pathology, counter, status in zip(pathologies, pathologies_counter, status_pathologies):
        print(f'{pathology}: {counter} bboxes - Benign({status.count("Benign")}) - Malignant({status.count("Malignant")})')

    negated_scans = []
    crop_coordinates = []
    if not Path(png_folder).exists():
        Path(png_folder).mkdir(parents=True)
    for group_name, df_group in unique_image_id_df:
        for image in df_group.itertuples():
            scan_nifti = os.path.join(nifti_folder, 'images')
            scan_nifti = os.path.join(scan_nifti, image.client_id)
            scan_nifti = os.path.join(scan_nifti, image.study_id)
            scan_nifti = os.path.join(scan_nifti, image.image_id + '.nii.gz')
            if Path(scan_nifti).exists():
                scan_png = os.path.join(png_folder, 'images')
                scan_png = os.path.join(scan_png, image.client_id)
                scan_png = os.path.join(scan_png, image.study_id)
                if not Path(scan_png).exists():
                    Path(scan_png).mkdir(parents=True)
                scan_png = os.path.join(scan_png, image.image_id + '.png')
                wrong_bboxes = ['1.2.826.0.1.3680043.9.3218.1.1.21149576.1882.1552598184295.155.0',
                              '1.2.826.0.1.3680043.9.3218.1.1.423368237.8725.1541832007115.79.0',
                              '1.2.826.0.1.3680043.9.3218.1.1.23278785.1708.1544221170622.213.0',
                              '1.2.826.0.1.3680043.9.3218.1.1.3010695.7588.1512129590473.9978.0',
                              '1.2.826.0.1.3680043.9.3218.1.1.242205684.1858.1540020381577.17.0'] # X and Y bbox flipped?
                #if image.image_id in wrong_bbox:
                if not os.path.exists(scan_png):
                #if image.image_id == test_id:
                    # Convert NIFTI to PNG
                    nii = nib.load(scan_nifti)
                    ndarray_nii = np.array(nii.get_fdata()).copy()
                    max_pix_value = ndarray_nii.max()
                    scaled_scan = cv2.convertScaleAbs(np.squeeze(ndarray_nii), alpha=(255.0/float(max(max_pix_value,1))))
                    bright_pix = np.count_nonzero(scaled_scan == 255)
                    total_pix = ndarray_nii.shape[0]*ndarray_nii.shape[1]
                    brigth_perc = bright_pix*100/total_pix
                    #cv2.imwrite('./test_0.png', np.transpose(scaled_scan, (1, 0)))
                    # plot_image_opencv_fit_window(np.transpose(scaled_scan, (1, 0)), 
                    #                     title='OPTIMAM Scan', screen_resolution=(1920, 1080),
                    #                     wait_key=True)
                    if (image.implant == 'NO' and brigth_perc > 25) or (image.implant == 'YES' and brigth_perc > 70):
                        scaled_scan = 255 - scaled_scan
                        negated_scans.append(scan_png)
                    png_image  = np.transpose(scaled_scan, (1, 0))
                    if crop_breast:
                        png_image = np.flipud(png_image)
                        png_image, (xmin, xmax, ymin, ymax) = fit_to_breast(png_image)
                        crop_coordinates.append([image.image_id, xmin, xmax, ymin, ymax])
                    else:
                        png_image = np.flipud(png_image)
                    with open(scan_png, 'wb') as png_file:
                        w = png.Writer(png_image.shape[1], png_image.shape[0], greyscale=True)
                        w.write(png_file, png_image.copy())
                    print(f'Saved {scan_png}')
                    img = np.array(Image.open(scan_png).convert('RGB'))
                    #cv2.imwrite('./test_1.png', img)
                    plot_image_opencv_fit_window(img, title='OPTIMAM Scan', screen_resolution=(1920, 1080),
                                        wait_key=True)
                break
    df = pd.DataFrame(negated_scans, columns=['neg_scans'])
    df.to_csv(os.path.join(png_folder, 'neg_scans.csv'), index=False)
    if crop_breast:
        df = pd.DataFrame(crop_coordinates, columns=['image_id', 'xmin', 'xmax', 'ymin', 'ymax'])
        df.to_csv(os.path.join(png_folder, 'crop_to_breast_coordinates.csv'), index=False) 

def plot_optimam_bboxes(info_csv, png_folder):

    pathologies = ['mass', 'calcifications', 'architectural_distortion', 'focal_asymmetry',
                    'suspicious_calcifications']
    info = pd.read_csv(info_csv)
    info = info.astype(object).replace(np.nan, '')
    info.head()
    # df = info.groupby(['image_id']).size().reset_index(name='counts')
    df_bbox = info.loc[info['mark_id'] != '']
    bbox_counts = df_bbox.groupby(['image_id']).size().reset_index(name='counts')
    print(f'Scans with annotated bboxes: {len(bbox_counts)}')
    pathologies_counter = []
    status_pathologies = []
    for pathology in pathologies:
        count = 0
        status = []
        for i in df_bbox.itertuples():
            if pathology in i.pathologies:
                if i.pathologies.count("suspicious") and i.pathologies.count("calcifications") < 2:
                    if pathology == 'suspicious_calcifications':
                        count += 1
                        status.append(i.status)
                else:
                    count += 1
                    status.append(i.status)
        pathologies_counter.append(count)
        status_pathologies.append(status)

    for pathology, counter, status in zip(pathologies, pathologies_counter, status_pathologies):
        print(f'{pathology}: {counter} bboxes - Benign({status.count("Benign")}) - Malignant({status.count("Malignant")})')

    unique_image_id_df = info.groupby(['image_id'], as_index=False)
    for group_name, df_group in unique_image_id_df:
        for idx_mark, image in enumerate(df_group.itertuples()):
            scan_png = os.path.join(png_folder, 'images')
            scan_png = os.path.join(scan_png, image.client_id)
            scan_png = os.path.join(scan_png, image.study_id)
            scan_png = os.path.join(scan_png, image.image_id + '.png')
            if os.path.exists(scan_png) and image.x1:
                # check_scan = '/home/lidia/Datasets/OPTIMAM/png_screening/images/demd2759/1.2.826.0.1.3680043.9.3218.1.1.1182262.1584.1511946747160.1457.0/1.2.826.0.1.3680043.9.3218.1.1.1182262.1584.1511946747160.1468.0.png'
                # if scan_png == check_scan:
                    # print(scan_png)
                print(image.pathologies)
                if idx_mark == 0:
                    img = np.array(Image.open(scan_png).convert('RGB'))
                    # img = cv2.flip(img, 0)
                img_height, img_width = img.shape[0], img.shape[1]
                # Vertically Mirroed Scans in PNG
                top, left = int(image.x1), int(image.y1)
                bottom, right = int(image.x2), int(image.y2)
                # top, left = int(image.x1), img_height - int(image.y1)
                # bottom, right = int(image.x2),  img_height - int(image.y2)
                cv2.rectangle(img, (top, left), (bottom, right), (0, 255, 0), 4)
                #define the screen resulation
                screen_res = 1920, 1080
                scale_width = screen_res[0] / img_width
                scale_height = screen_res[1] / img_height 
                scale = min(scale_width, scale_height)
                #resized window width and height
                window_width = int(img_width * scale)
                window_height = int(img_height * scale)
                if idx_mark == len(df_group)-1:
                    cv2.startWindowThread()
                    #cv2.WINDOW_NORMAL makes the output window resizealbe
                    cv2.namedWindow('OPTIMAM Scan', cv2.WINDOW_NORMAL)
                    #resize the window according to the screen resolution
                    cv2.resizeWindow('OPTIMAM Scan', window_width, window_height)
                    cv2.imshow('OPTIMAM Scan', img)
                    cv2.waitKey()

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

if __name__ == '__main__':

    if len(sys.argv) > 1:
        config_file = Path(sys.argv[1])
        chunks = int(sys.argv[2])
        first_run = int(sys.argv[3])
        counter = int(sys.argv[4])
    else:
        config_file = Path(
            '/home/lidia/source/BreastCancer/src/configs/preprocess_optimam.yaml')
        first_run = 0
        chunks = -1
        counter = 0
    with open(config_file) as file:
        config = yaml.safe_load(file)

    nifti_folder = config['paths']['nifti_folder']
    png_folder = config['paths']['png_folder']

    if chunks == -1:
        client_list = None
        if config['processing']['save_nifti']:
                    convert_optimam_dicom_to_nifti(config, client_list, nifti_folder, first_run=first_run)
        if config['processing']['save_png']:
            # insert_cropped_coordinates_to_info_csv(config['paths']['info_csv'], 
            #                                         config['paths']['cropped_coord'])
            convert_optimam_nifti_to_png(config['paths']['info_csv'], nifti_folder, png_folder,
                                        crop_breast=config['processing']['crop_breast'])
        if config['plot']['bbox']:
            plot_optimam_bboxes(config['paths']['info_csv'], png_folder)
        if config['plot']['negated']:
            neg_images = pd.read_csv(config['paths']['negated_images'])
            for image in neg_images.itertuples():
                pil_image = np.array(Image.open(image[1]).convert('RGB'))
                print(f'Showing {image[1]}')
                plot_image_opencv_fit_window(pil_image, screen_resolution=(1920, 1080),
                                            wait_key=True)
    else:
        dirname = os.path.join(config['paths']['dataset'], 'images')
        dirfiles = os.listdir(dirname)

        if first_run == 1:
            first_run = True
        else:
            first_run = False

        for idx, group in enumerate(chunker(dirfiles, chunks)):
            if idx == counter:
                client_list = group
                if config['processing']['save_nifti']:
                    convert_optimam_dicom_to_nifti(config, client_list, nifti_folder, first_run=first_run)
                    break
                if config['processing']['save_png']:
                    #db = omidb.DB(dataset_path, clients=['demd156'])
                    info_csv = config['paths']['info_csv']
                    convert_optimam_nifti_to_png(info_csv, nifti_folder, png_folder)
                    break

# Image with artifacts:
#'demd6778/1.2.826.0.1.3680043.9.3218.1.1.32172645.3148.1512150247410.742.0/1.2.826.0.1.3680043.9.3218.1.1.32172645.3148.1512150247410.748.0.png'