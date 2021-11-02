import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageOps
import cv2
# import matplotlib.pyplot as plt
import mmcv
import torch
from src.data_handling.mmg_detection_datasets import MANUFACTURERS_BCDR, PATHOLOGIES_BCDR, PIXEL_SIZE_BCDR

from src.preprocessing.histogram_standardization import get_hist_stand_landmarks, apply_hist_stand_landmarks
from src.preprocessing.histogram_standardization import plot_img_and_hist
import matplotlib.pyplot as plt
from src.data_handling.common_methods_mmg import *
from src.data_handling.mmg_detection_datasets import *

SEED = 999
np.random.seed(SEED)

def save_dataset_to_COCO(info_csv, dataset_path, cropped_to_breast, pathologies,
                                output_path, txt_out, json_out, category_id_dict):

    BCDR_clients = BCDRDataset(info_csv, dataset_path, detection=True, load_max=-1, 
                                cropped_to_breast=cropped_to_breast)
    # pathologies = PATHOLOGIES_BCDR
    if pathologies:
        clients_selected = BCDR_clients.get_clients_by_pathology_and_status(pathologies)
        print(f'Total clients in loaded dataset: {len(BCDR_clients)}')
        print(f'Pahologies selected: {pathologies}')
        print('-----------------------------------')
        print(f'Clients: {BCDR_clients.total_clients(pathologies)} - Images: {BCDR_clients.total_images(pathologies)} - Annotations: {BCDR_clients.total_annotations(pathologies)}')
    else:
        clients_selected =  BCDR_clients
    mass_status = {'Benign': 0, 'Malignant': 0, 'Normal': 0}
    breast_density = {0: 0, 1 :0, 2 :0, 3 :0, 4: 0}
    view = {'CC': 0, 'MLO' :0}
    age = {'<50': 0, '50 to 60': 0, '60 to 70': 0, '>70': 0}

    total_images = 0
    image_list = []
    image_ids = []
    for client in clients_selected:
        images = client.get_images_by_pathology(pathologies)
        total_images += len(images)
        for image in images:
            image_list.append(image.path)
            image_ids.append(image.id)
            if image.annotations:
                for annotation in image.annotations:
                    if any(item in annotation.pathologies for item in pathologies):
                        mass_status[image.status] += 1
                        breast_density[image.breast_density] += 1
                        view[image.view] += 1
                        if image.age < 50:
                            age['<50'] += 1
                        elif image.age >= 50 and image.age < 60:
                            age['50 to 60'] += 1
                        elif image.age >= 60 and image.age < 70:
                            age['60 to 70'] += 1
                        else:
                            age['>70'] += 1
            else:
                mass_status[image.status] += 1
                breast_density[image.breast_density] += 1
                view[image.view] += 1
                if image.age < 50:
                    age['<50'] += 1
                elif image.age >= 50 and image.age < 60:
                    age['50 to 60'] += 1
                elif image.age >= 60 and image.age < 70:
                    age['60 to 70'] += 1
                else:
                    age['>70'] += 1

    print(total_images)
    print(f'Images by BI-RAD: {mass_status}')
    print(f'Images by Breast Density (ACR): {breast_density}')
    print(f'Images by View: {view}')
    print(f'Images by Age: {age}')

    duplicates = set([x for x in image_list if image_list.count(x) > 1])
    non_duplicates = list(set(image_list))
    duplicates_ids = set([x for x in image_ids if image_ids.count(x) > 1])

    #save_images_paths(clients_selected, os.path.join(output_path, txt_out))
    save_images_ids(clients_selected, os.path.join(output_path, txt_out))
    json_file = os.path.join(output_path, json_out)
    bbox_ctr = 0
    df_bboxes = pd.DataFrame(columns=['image', 'bbox_width', 'bbox_height'])
    clients_save_COCO_annotations(clients_selected, pathologies, json_file, fit_to_breast=fit_to_breast,
                                    category_id_dict=category_id_dict, use_status=use_status,
                                    df_bboxes=df_bboxes)
    df_bboxes.to_csv(os.path.join(output_path, f'BCDR_{class_name}_rescaled_bboxes.csv'), index=False)

def save_dataset_split_to_COCO(info_csv, dataset_path, cropped_to_breast, pathologies, client_ids,
                                output_path, txt_out, json_out, category_id_dict):
    BCDR_clients = BCDRDataset(info_csv, dataset_path, detection=True, load_max=-1, 
                            cropped_to_breast=cropped_to_breast, client_ids=client_ids)
    clients_selected = BCDR_clients.get_clients_by_pathology_and_status(pathologies)
    print(f'Total clients in loaded dataset: {len(BCDR)}')
    print(f'Pahologies selected: {pathologies}')
    print('-----------------------------------')
    print(f'Clients: {BCDR_clients.total_clients(pathologies)} - Images: {BCDR_clients.total_images(pathologies)} - Annotations: {BCDR_clients.total_annotations(pathologies)}')
    save_images_paths(clients_selected, os.path.join(output_path, txt_out))
    json_file = os.path.join(output_path, json_out)
    clients_save_COCO_annotations(clients_selected, pathologies, json_file, fit_to_breast=fit_to_breast,
                                    category_id_dict=category_id_dict, use_status=use_status)

def clients_save_COCO_annotations(clients, pathologies, json_file, fit_to_breast=False,
                                  category_id_dict={'Benign_mass': 0, 'Malignant_mass': 1},
                                  use_status=False, df_bboxes=None):
    annotations = []
    images_dict = []
    obj_count = 0
    image_id = 0
    bbox_count = 0
    if pathologies:
        for client in clients:
            images = client.get_images_by_pathology(pathologies)
            for image in images:
                img_elem, annot_elem, obj_count = image.generate_COCO_dict(image_id, obj_count, pathologies,
                                                        fit_to_breast=fit_to_breast, category_id_dict=category_id_dict,
                                                        use_status=use_status)
                images_dict.append(img_elem)
                if df_bboxes is not None:
                    # Resize image keeping aspect ratio
                    scale = (1333, 800)
                    img = np.array(Image.open(img_elem['file_name']))
                    h, w = img.shape[:2]
                    # h, w = img_elem['height'], img_elem['width']
                    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
                    rescaled_img, w_scale, h_scale = imresize(img, new_size, return_scale=True, backend='pillow')
                    scale_factor_bbox = np.array([w_scale, h_scale, w_scale, h_scale],
                                        dtype=np.float32)
                for annotation in annot_elem:
                    annotations.append(annotation)
                    if df_bboxes is not None:
                        rescaled_bbox = annotation['bbox'] * scale_factor_bbox
                        rescaled_bbox = rescaled_bbox.astype(int)
                        df_bboxes.loc[bbox_count] = [image.id] + [img_elem['file_name']] + [annotation['bbox'][2]*image.pixel_spacing] + \
                                                    [annotation['bbox'][3]*image.pixel_spacing] + [rescaled_bbox[2]] + [rescaled_bbox[3]] + \
                                                    [image.view] + [image.status] + [image.breast_density] + [image.age]
                        bbox_count += 1
                    if False:
                        xmax = rescaled_bbox[2] + rescaled_bbox[0]
                        ymax = rescaled_bbox[3] + rescaled_bbox[1]
                        poly = [rescaled_bbox[0], rescaled_bbox[1]], [int((rescaled_bbox[2])/2) + rescaled_bbox[0], rescaled_bbox[1]], \
                            [xmax, rescaled_bbox[1]], [xmax, int((rescaled_bbox[3])/2) + rescaled_bbox[1]], \
                            [xmax, ymax], [int((rescaled_bbox[2])/2) + rescaled_bbox[0], ymax], \
                            [rescaled_bbox[0], ymax], [rescaled_bbox[0], int((rescaled_bbox[3])/2) + rescaled_bbox[1]]
                        img = cv2.cvtColor(rescaled_img, cv2.COLOR_BGR2RGB)
                        cv2.polylines(img, [np.array(poly)], True, (0, 255, 0), 2)
                        plt.figure()
                        plt.imshow(img)
                        plt.show()
                image_id += 1
    else:
        for client in clients:
            for study in client:
                for image in study:
                    img_elem, annot_elem, obj_count = image.generate_COCO_dict(image_id, obj_count, pathologies,
                                                            fit_to_breast=fit_to_breast, category_id_dict=category_id_dict,
                                                            use_status=use_status)
                    images_dict.append(img_elem)
                    for annotation in annot_elem:
                        annotations.append(annotation)
                    image_id += 1
    print(f'Total images: {image_id} - total objects: {obj_count}')
    # Build categories dict
    categories = []
    for item in category_id_dict.items():
        categories.append({'id': item[1], 'name': item[0]})
    coco_format_json = dict(
                            images=images_dict,
                            annotations=annotations,
                            categories=categories)
    if json_file:
        mmcv.dump(coco_format_json, json_file)
    if df_bboxes is not None:
        return df_bboxes

# def save_dataset_abnormal_classification(BCDR_dataset, pathologies, manufacturer, output_path, 
#             out_prefix='BCDR_abnormal_masses'):
        
#         normal_clients = BCDR_dataset.get_clients_by_pathology_and_status(pathologies=None, status=['Normal'])
#         normal_image_paths = []
#         if not Path(os.path.join(output_path, 'BCDR_normal_image_list.txt')).exists():
#             for client in normal_clients:
#                 normal_images = client.get_images_by_status(status=['Normal'])
#                 for image in normal_images:
#                     if image.manufacturer == manufacturer and not image.implant:
#                         normal_image_paths.append(image.path)
#             with open(os.path.join(output_path, 'BCDR_normal_image_list.txt'), 'w') as f:
#                 f.write("\n".join(normal_image_paths))

#         abnormal_clients = BCDR_dataset.get_clients_by_pathology_and_status(pathologies=pathologies)
#         abnormal_image_paths = []
#         for client in abnormal_clients:
#             abnormal_images = client.get_images_by_pathology(pathologies)
#             for image in abnormal_images:
#                 if image.manufacturer == manufacturer and not image.implant:
#                     abnormal_image_paths.append(image.path)
#         with open(os.path.join(output_path, f'{out_prefix}_image_list.txt'), 'w') as f:
#             f.write("\n".join(abnormal_image_paths))

def get_bbox_size_distribution(info_csv, dataset_path, detection, cropped_to_breast,
                            pathologies, image_list=None, out_csv=None):
    if image_list:
        image_ids = []
        image_paths = open(image_list,'r').read().split('\n')
        for path in image_paths:
            image_ids.append(path)
        dataloader = BCDRDataset(info_csv, dataset_path,
                        detection=detection, load_max=-1, image_ids=image_ids, 
                        cropped_to_breast=cropped_to_breast)
    else:
        dataloader = BCDRDataset(info_csv, dataset_path, detection=detection, load_max=-1, 
                                    cropped_to_breast=cropped_to_breast)
    
    clients_selected = dataloader.get_clients_by_pathology_and_status(pathologies)
    print(f'Pahologies selected: {pathologies}')
    print('-----------------------------------')
    print(f'Clients: {dataloader.total_clients(pathologies)} - Images: {dataloader.total_images(pathologies)} - Annotations: {dataloader.total_annotations(pathologies)}')
    #df_bboxes = pd.DataFrame(columns=['image', 'bbox_width_mm', 'bbox_height_mm', 'bbox_width_rescaled', 'bbox_height_rescaled'])
    df_bboxes = pd.DataFrame(columns=['image_id', 'image_path', 'bbox_width_mm', 'bbox_height_mm', 'bbox_width_rescaled', 'bbox_height_rescaled', 
                                        'view', 'status', 'breast_density', 'age'])
    df_bboxes = clients_save_COCO_annotations(clients_selected, pathologies, json_file=None, fit_to_breast=fit_to_breast,
                            category_id_dict=category_id_dict, use_status=use_status,
                            df_bboxes=df_bboxes)
    if out_csv:
        df_bboxes.to_csv(out_csv, index=False)
    
    total_info = {'total': 0,
                 'status': {'Benign': 0, 'Malignant': 0},
                 'view': {'CC': 0, 'MLO' :0},
                 'breast_density': {1 :0, 2 :0, 3 :0, 4: 0},
                 'age': {'<50': 0, '50 to 60': 0, '60 to 70': 0, '>70': 0}}
    diameter_info = {'<1.0cm': {'total':0, 'status':{'Benign': 0, 'Malignant': 0}, 'view': {'CC': 0, 'MLO' :0}, 'breast_density': {1 :0, 2 :0, 3 :0, 4: 0}, 'age': {'<50': 0, '50 to 60': 0, '60 to 70': 0, '>70': 0}},
                '1.0-1.5cm': {'total':0, 'status':{'Benign': 0, 'Malignant': 0}, 'view': {'CC': 0, 'MLO' :0}, 'breast_density': {1 :0, 2 :0, 3 :0, 4: 0}, 'age': {'<50': 0, '50 to 60': 0, '60 to 70': 0, '>70': 0}},
                '1.5-2.0cm': {'total':0, 'status':{'Benign': 0, 'Malignant': 0}, 'view': {'CC': 0, 'MLO' :0}, 'breast_density': {1 :0, 2 :0, 3 :0, 4: 0}, 'age': {'<50': 0, '50 to 60': 0, '60 to 70': 0, '>70': 0}},
                '2.0-3.0cm': {'total':0, 'status':{'Benign': 0, 'Malignant': 0}, 'view': {'CC': 0, 'MLO' :0}, 'breast_density': {1 :0, 2 :0, 3 :0, 4: 0}, 'age': {'<50': 0, '50 to 60': 0, '60 to 70': 0, '>70': 0}},
                '3.0-4.0cm': {'total':0, 'status':{'Benign': 0, 'Malignant': 0}, 'view': {'CC': 0, 'MLO' :0}, 'breast_density': {1 :0, 2 :0, 3 :0, 4: 0}, 'age': {'<50': 0, '50 to 60': 0, '60 to 70': 0, '>70': 0}},
                '>4.0cm': {'total':0, 'status':{'Benign': 0, 'Malignant': 0}, 'view': {'CC': 0, 'MLO' :0}, 'breast_density': {1 :0, 2 :0, 3 :0, 4: 0}, 'age': {'<50': 0, '50 to 60': 0, '60 to 70': 0, '>70': 0}}}
    df = pd.read_csv(out_csv)
    for index, row in df.iterrows():
        diameter_mm = int(max(row['bbox_width_mm'], row['bbox_height_mm']))
        if diameter_mm <= 10:
            d_str = '<1.0cm'
        elif diameter_mm > 10 and diameter_mm <= 15:
            d_str = '1.0-1.5cm'
        elif diameter_mm > 15 and diameter_mm <= 20:
            d_str = '1.5-2.0cm'
        elif diameter_mm > 20 and diameter_mm <= 30:
            d_str = '2.0-3.0cm'
        elif diameter_mm > 30 and diameter_mm <= 40:
            d_str = '3.0-4.0cm'
        elif diameter_mm > 40:
            d_str = '>4.0cm'
        total_info['total'] += 1
        diameter_info[d_str]['total'] += 1
        if row['breast_density'] > 0:
            diameter_info[d_str]['breast_density'][row['breast_density']] += 1
            total_info['breast_density'][row['breast_density']] += 1
        diameter_info[d_str]['view'][row['view']] += 1
        total_info['view'][row['view']] += 1
        diameter_info[d_str]['status'][row['status']] += 1
        total_info['status'][row['status']] += 1
        if row['age'] < 50:
            diameter_info[d_str]['age']['<50'] += 1
            total_info['age']['<50'] += 1
        elif row['age'] >= 50 and row['age'] < 60:
            diameter_info[d_str]['age']['50 to 60'] += 1
            total_info['age']['50 to 60'] += 1
        elif row['age'] >= 60 and row['age'] < 70:
            diameter_info[d_str]['age']['60 to 70'] += 1
            total_info['age']['60 to 70'] += 1
        else:
            diameter_info[d_str]['age']['>70'] += 1
            total_info['age']['>70'] += 1
    
    s = pd.Series(diameter_info,index=diameter_info.keys())
    info_s = pd.Series(total_info,index=total_info.keys())
    df_out = pd.DataFrame(columns=['mass_diameter', 'total', 'Benign', 'Malignant', 'MLO', 'CC', 'ACR1', 'ACR2', 'ACR3', 'ACR4', '<50', '50 to 60', '50 to 60', '>70'])
    row_ctr = 0
    for index, value in s.items():
        row_v = []
        for index_total, value_total in info_s.items():
            if index_total == 'total':
                val = value[index_total]
                cent = int(np.round(100*val/value_total))
                row_v = row_v + [f'{cent}% / {val}']
            else:
                for key in value_total.keys():
                    val = value[index_total][key]
                    cent = int(np.round(100*val/value_total[key]))
                    row_v = row_v + [f'{cent}% / {val}']
        df_out.loc[row_ctr] = [index] + row_v
        row_ctr += 1
    return df_out

def get_dataset_distribution(info_csv, dataset_path, detection, cropped_to_breast,
                            pathologies, image_list=None):
    
    if image_list:
        image_ids = []
        image_paths = open(image_list,'r').read().split('\n')
        for path in image_paths:
            image_ids.append(path)
        dataloader = BCDRDataset(info_csv, dataset_path,
                        detection=detection, load_max=-1, image_ids=image_ids, 
                        cropped_to_breast=cropped_to_breast)
    else:
        dataloader = BCDRDataset(info_csv, dataset_path, detection=detection, load_max=-1, 
                                    cropped_to_breast=cropped_to_breast)
    
    clients_selected = dataloader.get_clients_by_pathology_and_status(pathologies)
    print(f'Pahologies selected: {pathologies}')
    print('-----------------------------------')
    print(f'Clients: {dataloader.total_clients(pathologies)} - Images: {dataloader.total_images(pathologies)} - Annotations: {dataloader.total_annotations(pathologies)}')
    
    mass_status = {'Benign': 0, 'Malignant': 0, 'Normal': 0}
    breast_density = {0: 0, 1 :0, 2 :0, 3 :0, 4: 0}
    view = {'CC': 0, 'MLO' :0}
    age = {'<50': 0, '50 to 60': 0, '60 to 70': 0, '>70': 0}

    total_images = 0
    image_list = []
    image_ids = []
    for client in clients_selected:
        images = client.get_images_by_pathology(pathologies)
        total_images += len(images)
        for image in images:
            image_list.append(image.path)
            image_ids.append(image.id)
            if image.annotations:
                for annotation in image.annotations:
                    if any(item in annotation.pathologies for item in pathologies):
                        mass_status[image.status] += 1
                        breast_density[image.breast_density] += 1
                        view[image.view] += 1
                        if image.age < 50:
                            age['<50'] += 1
                        elif image.age >= 50 and image.age < 60:
                            age['50 to 60'] += 1
                        elif image.age >= 60 and image.age < 70:
                            age['60 to 70'] += 1
                        else:
                            age['>70'] += 1
            else:
                mass_status[image.status] += 1
                breast_density[image.breast_density] += 1
                view[image.view] += 1
                if image.age < 50:
                    age['<50'] += 1
                elif image.age >= 50 and image.age < 60:
                    age['50 to 60'] += 1
                elif image.age >= 60 and image.age < 70:
                    age['60 to 70'] += 1
                else:
                    age['>70'] += 1

    print(total_images)
    print(f'Images by BI-RAD: {mass_status}')
    print(f'Images by Breast Density (ACR): {breast_density}')
    print(f'Images by View: {view}')
    print(f'Images by Age: {age}')

    # duplicates = set([x for x in image_list if image_list.count(x) > 1])
    # non_duplicates = list(set(image_list))
    # duplicates_ids = set([x for x in image_ids if image_ids.count(x) > 1])

from src.visualizations.plot_image import plot_image_opencv_fit_window
if __name__ == '__main__':

    # Cropped scans
    info_csv= ['/home/lidia/Datasets/BCDR/cropped/BCDR-D01_dataset/dataset_info.csv',
                '/home/lidia/Datasets/BCDR/cropped/BCDR-D02_dataset/dataset_info.csv']
    dataset_path= ['/home/lidia/Datasets/BCDR/cropped/BCDR-D01_dataset',
                    '/home/lidia/Datasets/BCDR/cropped/BCDR-D02_dataset']
    output_path = '/home/lidia/Datasets/BCDR/cropped/detection/masses'

    # Cropped scans GPU Server
    # info_csv= ['/home/lidia-garrucho/datasets/BCDR/cropped/BCDR-D01_dataset/dataset_info.csv',
    #             '/home/lidia-garrucho/datasets/BCDR/cropped/BCDR-D02_dataset/dataset_info.csv']
    # dataset_path= ['/home/lidia-garrucho/datasets/BCDR/cropped/BCDR-D01_dataset',
    #                 '/home/lidia-garrucho/datasets/BCDR/cropped/BCDR-D02_dataset']
    # output_path = '/home/lidia-garrucho/datasets/BCDR/cropped/detection/masses'

    # Save resized images 
    # info_csv= ['/home/lidia/Datasets/BCDR/cropped/BCDR-DN01_dataset/dataset_info.csv']
    # dataset_path= ['/home/lidia/Datasets/BCDR/cropped/BCDR-DN01_dataset']
    # output_path = '/home/lidia/Datasets/BCDR/style_transfer/1333_800/testB'
    # info_csv= ['/home/lidia/Datasets/BCDR/cropped/BCDR-D01_dataset/dataset_info.csv',
    #             '/home/lidia/Datasets/BCDR/cropped/BCDR-D02_dataset/dataset_info.csv']
    # dataset_path= ['/home/lidia/Datasets/BCDR/cropped/BCDR-D01_dataset',
    #                 '/home/lidia/Datasets/BCDR/cropped/BCDR-D02_dataset']
    # output_path = '/home/lidia/Datasets/BCDR/style_transfer/1333_800/testA'
    selected_breast_density = [4] #[0,1,2]
    out_height = 1333
    out_width = 800

    cropped_to_breast = True
    fit_to_breast = True

    plot_BCDR = False
    plot_studies = False
    # Compute Landmarks for Gistogram Standardisation
    compute_train_landmarks = False
    test_landmarks = False
    landmarks_name = 'BCDR_train_landmarks_mass'

    save_COCO = False
    # PATHOLOGIES = ['nodule', 'calcification', 'microcalcification', 'axillary_adenopathy',
    #             'architectural_distortion', 'stroma_distortion']
    class_name = 'mass'
    pathologies = ['mass'] #None
    use_status = False
    category_id_dict = {'mass': 0}

    detection = True
    load_max = -1
    if not Path(output_path).exists():
        Path(output_path).mkdir(parents=True)
    
    image_list = '/home/lidia/Datasets/BCDR/cropped/detection/masses/BCDR_mass_ids_all_digital.txt'
    out_csv = os.path.join(output_path, f'BCDR_mass_digital_all_bboxes.csv')
    diameter_info = get_bbox_size_distribution(info_csv, dataset_path, detection, cropped_to_breast,
                                    pathologies, image_list=image_list, out_csv=out_csv)
    out_bbox_folder = '/home/lidia/source/mmdetection/experiments/GPU_server/experiments/plots/bbox_size_dist'
    diameter_info.to_csv(os.path.join(out_bbox_folder, f'BCDR_bboxes_size_info.csv'), index=False)
    get_dataset_distribution(info_csv, dataset_path, detection, cropped_to_breast,
                            pathologies, image_list=image_list)

    BCDR = BCDRDataset(info_csv, dataset_path, detection=detection, load_max=load_max, 
                            cropped_to_breast=cropped_to_breast)
    print(f'Total clients in loaded dataset: {len(BCDR)}')
    print(f'Pahologies selected: {pathologies}')
    print('-----------------------------------')
    print(f'Clients: {BCDR.total_clients(pathologies)} - Images: {BCDR.total_images(pathologies)} - Annotations: {BCDR.total_annotations(pathologies)}')


    #data_BCDR = BCDR
    if pathologies is None:
        data_BCDR = BCDR
    else:
        clients_pathology = BCDR.get_clients_by_pathology_and_status(pathologies)
        data_BCDR = clients_pathology

    if plot_BCDR:
        breast_densities = [1]
        if plot_studies:
            for client in data_BCDR:
                if client.breast_density in breast_densities:
                    for study in client:
                        print(f'Client ID: {client.id}')
                        study.plot_study(print_annotations=True, fit_to_breast=fit_to_breast)
        else:
            # Plot dataset
            BCDR.plot_dataset(print_annotations=True, fit_to_breast=fit_to_breast, max=load_max)

    # Save full datastet to COCO
    save_dataset_to_COCO(info_csv, dataset_path, cropped_to_breast, pathologies,
                        output_path=output_path, txt_out=f'BCDR_{class_name}_ids_all_digital.txt',
                        json_out=f'BCDR_{class_name}_ids_all_digital.json', category_id_dict=category_id_dict)

    # Save resized images 
    mass_status = {'Benign': 0, 'Malignant': 0, 'Normal': 0}
    breast_density = {0: 0, 1 :0, 2 :0, 3 :0, 4: 0}
    for client in data_BCDR:
        for study in client.studies:
            images = study.get_images_by_pathology(pathologies)
            for image in images:
                lesion_status = image.status
                mass_status[lesion_status] += 1
                breast_density[image.breast_density] += 1
                if image.breast_density in selected_breast_density:
                    image_filename = str(client.id) + '_' + str(study.id) + '_' + os.path.basename(image.path)
                    resized_image_path = os.path.join(output_path, image_filename)
                    #Resize keeping aspect ratio
                    image = Image.open(image.path)
                    image.thumbnail([out_width, out_height], Image.ANTIALIAS)
                    delta_w = out_width - image.size[0]
                    delta_h = out_height - image.size[1]
                    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
                    new_im = ImageOps.expand(image, padding)
                    new_im.save(resized_image_path, "PNG")

    print(f'Clients with {pathologies}: {len(data_BCDR)}')
    print(f'Images by biopsy result: {mass_status}')
    print(f'Images by Breast Density  (ACR): {breast_density}')
    save_images_paths(clients_pathology, os.path.join(output_path, 'BCDR_mass_image_list.txt'))

    # Train, val, test splits
    X = []
    y = []
    for client in data_BCDR:
        images = client.get_images_by_pathology(pathologies)
        for image in images:
            X.append(client.id)
            y.append(image.breast_density)
            break
    train_split = 0.7
    val_split = 0.1
    test_split = 0.2
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1-train_split, stratify=y, random_state=SEED, shuffle=True)
    relative_frac_test = test_split / (val_split + test_split)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=relative_frac_test, stratify=y_temp, random_state=SEED, shuffle=True)
    assert len(X) == len(X_train) + len(X_val) + len(X_test)

    if test_landmarks:
        BCDR_test = BCDRDataset(info_csv, dataset_path, detection=True, load_max=-1, 
                                cropped_to_breast=cropped_to_breast, client_ids=X_test)
        test_clients = BCDR.get_clients_by_pathology_and_status(pathologies)
        landmarks_path = os.path.join(output_path, f'{landmarks_name}.pth')
        landmarks_values = torch.load(landmarks_path)
        for client in test_clients:
            for study in client:
                for image in study:
                    if image.site == 'adde':
                        #input_img = np.array(Image.open(image.path).convert('RGB'))
                        input_img = np.array(Image.open(image.path))
                        # plot_image_opencv_fit_window(input_img, title='Input Image', 
                        #                          screen_resolution=(1920, 1080), wait_key=True)
                        nyul_img = apply_hist_stand_landmarks(Image.open(image.path), landmarks_values)
                        nyul_img = nyul_img.astype(np.uint8)
                        # plot_image_opencv_fit_window(new_img, title='Std Test', 
                        #                          screen_resolution=(1920, 1080), wait_key=True)
                        from skimage import exposure
                        # Contrast stretching
                        plow, phigh = np.percentile(input_img, (0.01, 99.9))
                        img_rescale = exposure.rescale_intensity(input_img, in_range=(plow, phigh))

                        plow, phigh = np.percentile(nyul_img, (2, 98))
                        img_nyul_strecth = exposure.rescale_intensity(nyul_img, in_range=(plow, phigh))
                        rgb = np.dstack((img_rescale,nyul_img,img_nyul_strecth))

                        # Display results
                        fig = plt.figure(figsize=(8, 5))
                        axes = np.zeros((2, 5), dtype=np.object)
                        axes[0, 0] = fig.add_subplot(2, 5, 1)
                        for i in range(1, 5):
                            axes[0, i] = fig.add_subplot(2, 5, 1+i, sharex=axes[0,0], sharey=axes[0,0])
                        for i in range(0, 5):
                            axes[1, i] = fig.add_subplot(2, 5, 6+i)

                        ax_img, ax_hist, ax_cdf = plot_img_and_hist(input_img, axes[:, 0])
                        ax_img.set_title('Input image')

                        y_min, y_max = ax_hist.get_ylim()
                        ax_hist.set_ylabel('Number of pixels')
                        ax_hist.set_yticks(np.linspace(0, y_max, 5))

                        ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
                        ax_img.set_title('Contrast stretching')

                        ax_img, ax_hist, ax_cdf = plot_img_and_hist(nyul_img, axes[:, 2])
                        ax_img.set_title('Std Nyul')

                        ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_nyul_strecth, axes[:, 3])
                        ax_img.set_title('Std Nyul + Stretching')

                        ax_img, ax_hist, ax_cdf = plot_img_and_hist(rgb, axes[:, 4])
                        ax_img.set_title('RGB')

                        ax_cdf.set_ylabel('Fraction of total intensity')
                        ax_cdf.set_yticks(np.linspace(0, 1, 5))

                        # prevent overlap of y-axis labels
                        fig.tight_layout()
                        plt.show()
        
    if save_COCO:
        
        print('Train Split')
        print('------------------------')
        save_dataset_split_to_COCO(info_csv, dataset_path, cropped_to_breast, pathologies, client_ids=X_train,
                                    output_path=output_path, txt_out=f'BCDR_{class_name}_train.txt',
                                    json_out=f'BCDR_{class_name}_train.json', category_id_dict=category_id_dict)
        print()
        print('Val Split')
        print('------------------------')
        save_dataset_split_to_COCO(info_csv, dataset_path, cropped_to_breast, pathologies, client_ids=X_val,
                                    output_path=output_path, txt_out=f'BCDR_{class_name}_val.txt',
                                    json_out=f'BCDR_{class_name}_val.json', category_id_dict=category_id_dict)
        print()
        print('Test Split')
        print('------------------------')
        save_dataset_split_to_COCO(info_csv, dataset_path, cropped_to_breast, pathologies, client_ids=X_test,
                                    output_path=output_path, txt_out=f'BCDR_{class_name}_test.txt',
                                    json_out=f'BCDR_{class_name}_test.json', category_id_dict=category_id_dict)
