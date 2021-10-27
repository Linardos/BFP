import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import mmcv
import torch

from src.preprocessing.histogram_standardization import get_hist_stand_landmarks, apply_hist_stand_landmarks
from src.preprocessing.histogram_standardization import plot_img_and_hist
from src.data_handling.common_methods_mmg import *
from src.data_handling.mmg_detection_datasets import *

SEED = 999
np.random.seed(SEED)

def save_dataset_split_to_COCO(info_csv, dataset_path, cropped_to_breast, pathologies, manufacturer, client_ids,
                                output_path, txt_out, json_out, category_id_dict,
                                data_aug_path=None, data_aug_info = None):
        optimam_clients = OPTIMAMDataset(info_csv, dataset_path, detection=True, load_max=-1, 
                                cropped_to_breast=cropped_to_breast, client_ids=client_ids)
        clients_selected = optimam_clients.get_clients_by_pathology_and_status(pathologies)
        print(f'Total clients in loaded dataset: {len(optimam)}')
        print(f'Pahologies selected: {pathologies}')
        print('-----------------------------------')
        print(f'Clients: {optimam_clients.total_clients(pathologies)} - Images: {optimam_clients.total_images(pathologies)} - Annotations: {optimam_clients.total_annotations(pathologies)}')
        # Check images in HOLOGIC manufacurer
        images_sites = {'adde': 0, 
                         'jarv': 0,
                         'stge': 0}
        mass_site_distribution = {'adde': {'Malignant' :0, 'Benign': 0}, 
                'jarv': {'Malignant' :0, 'Benign': 0},
                'stge': {'Malignant' :0, 'Benign': 0}}
        mass_conspicuity = {'Obvious' :0, 'Occult' :0, 'Subtle' :0, 'Very_subtle' :0, None: 0}
        pathologies_summary = {'mass' :0, 'calcification' :0, 'distortion' :0}
        for client in clients_selected:
            images = client.get_images_by_pathology(pathologies)
            for image in images:
                if image.manufacturer == manufacturer:
                    images_sites[image.site] += 1
                    total_masses = image.total_annotations(pathologies)
                    mass_site_distribution[image.site][image.status] += total_masses
                    for annotation in image.annotations:
                        if any(item in annotation.pathologies for item in pathologies):
                            mass_conspicuity[annotation.conspicuity] += 1
                            if 'mass' in annotation.pathologies and 'mass' in category_id_dict.keys():
                                pathologies_summary['mass'] += 1
                            elif ('calcifications' in annotation.pathologies or 'suspicious_calcifications' in annotation.pathologies) and 'calcification' in category_id_dict.keys():
                                pathologies_summary['calcification'] += 1
                            elif 'architectural_distortion' in annotation.pathologies and 'distortion' in category_id_dict.keys():
                                pathologies_summary['distortion'] += 1
                #         else:
                #             print(annotation.pathologies)
                # else:
                #     print(image.manufacturer)
        print(images_sites)
        print(mass_site_distribution)
        print(mass_conspicuity)
        print(pathologies_summary)
        save_images_paths(clients_selected, os.path.join(output_path, txt_out))
        if json_out:
            json_file = os.path.join(output_path, json_out)
            df_bboxes_rescaled = pd.DataFrame(columns=['image', 'bbox_width', 'bbox_height'])
            df_bboxes_rescaled = clients_save_COCO_annotations(clients_selected, pathologies, json_file, fit_to_breast=fit_to_breast,
                                    category_id_dict=category_id_dict, use_status=use_status,
                                    data_aug_path=data_aug_path, data_aug_info=data_aug_info,
                                    df_bboxes_rescaled=df_bboxes_rescaled)
            df_bboxes_rescaled.to_csv(os.path.join(output_path, f'OPTIMAM_train_split_{class_name}_rescaled_bboxes.csv'), index=False)

def clients_save_COCO_annotations(clients, pathologies, json_file, fit_to_breast=False,
                                  category_id_dict={'Benign_mass': 0, 'Malignant_mass': 1},
                                  use_status=False, data_aug_path=None, data_aug_info=None,
                                  df_bboxes_rescaled=None):
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
                if df_bboxes_rescaled is not None:
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
                    if df_bboxes_rescaled is not None:
                        rescaled_bbox = annotation['bbox'] * scale_factor_bbox
                        rescaled_bbox = rescaled_bbox.astype(int)
                        df_bboxes_rescaled.loc[bbox_count] = [img_elem['file_name']] + [rescaled_bbox[2]] + [rescaled_bbox[3]]
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
                if data_aug_path and data_aug_info:
                    img_elem, annot_elem, obj_count = image.generate_COCO_dict_high_density(data_aug_path, data_aug_info, 
                                                            image_id, obj_count, pathologies,
                                                            fit_to_breast=fit_to_breast, category_id_dict=category_id_dict,
                                                            use_status=use_status)
                    images_dict.append(img_elem)
                    for annotation in annot_elem:
                        annotations.append(annotation)
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
                    if data_aug_path and data_aug_info:
                        img_elem, annot_elem, obj_count = image.generate_COCO_dict_high_density(data_aug_path, data_aug_info,
                                                            image_id, obj_count, pathologies,
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
    mmcv.dump(coco_format_json, json_file)
    if df_bboxes_rescaled is not None:
        return df_bboxes_rescaled

def save_dataset_abnormal_classification(optimam_dataset, pathologies, manufacturer, output_path, 
            out_prefix='OPTIMAM_abnormal_masses'):
        
        normal_clients = optimam_dataset.get_clients_by_pathology_and_status(pathologies=None, status=['Normal'])
        normal_image_paths = []
        if not Path(os.path.join(output_path, 'OPTIMAM_normal_image_list.txt')).exists():
            for client in normal_clients:
                normal_images = client.get_images_by_status(status=['Normal'])
                for image in normal_images:
                    if image.manufacturer == manufacturer and not image.implant:
                        normal_image_paths.append(image.path)
            with open(os.path.join(output_path, 'OPTIMAM_normal_image_list.txt'), 'w') as f:
                f.write("\n".join(normal_image_paths))

        abnormal_clients = optimam_dataset.get_clients_by_pathology_and_status(pathologies=pathologies)
        abnormal_image_paths = []
        for client in abnormal_clients:
            abnormal_images = client.get_images_by_pathology(pathologies)
            for image in abnormal_images:
                if image.manufacturer == manufacturer and not image.implant:
                    abnormal_image_paths.append(image.path)
        with open(os.path.join(output_path, f'{out_prefix}_image_list.txt'), 'w') as f:
            f.write("\n".join(abnormal_image_paths))

def get_dataset_distribution(info_csv, dataset_path, detection, cropped_to_breast,
                            pathologies, image_list=None, manufacturers=None):
    
    if not manufacturers:
        manufacturers = MANUFACTURERS_OPTIMAM
    elif not isinstance(manufacturers, list):
        manufacturers = [manufacturers]

    if image_list:
        if not isinstance(image_list, list):
            image_list = [image_list]
        image_ids = []
        for i_list in image_list:
            image_paths = open(i_list,'r').read().split('\n')
            for path in image_paths:
                image_id = os.path.splitext(os.path.basename(path))[0]
                if image_id in WRONG_BBOXES_OPTIMAM:
                    continue
                image_ids.append(image_id)
        with Timer('Init OPTIMAM Dataset'):
            dataloader = OPTIMAMDataset(info_csv, dataset_path,
                            detection=detection, load_max=-1, image_ids=image_ids, 
                            cropped_to_breast=cropped_to_breast)
    else:
        dataloader = OPTIMAMDataset(info_csv, dataset_path, detection=detection, load_max=-1, 
                                    cropped_to_breast=cropped_to_breast)
        dataloader = dataloader.get_clients_by_pathology_and_status(pathologies)

    print(f'Pahologies selected: {pathologies}')
    print('-----------------------------------')
    print(f'Clients: {dataloader.total_clients(pathologies)} - Images: {dataloader.total_images(pathologies)} - Annotations: {dataloader.total_annotations(pathologies)}')
    
    mass_status = {'Benign': 0, 'Malignant': 0, 'Normal': 0}
    # breast_density = {0: 0, 1 :0, 2 :0, 3 :0, 4: 0}
    view = {'CC': 0, 'MLO' :0}
    age = {'<50': 0, '50 to 60': 0, '60 to 70': 0, '>70': 0}
    site = {'adde': 0, 'jarv': 0, 'stge': 0}
    total_clients = 0
    total_images = 0
    total_annotations = 0
    image_list = []
    image_ids = []
    for client in dataloader:
        images = client.get_images_by_pathology(pathologies)
        for image in images:
            if image.manufacturer in manufacturers:
                total_clients +=1
                break
        for image in images:
            if image.view not in CC_VIEWS_OPTIMAM + MLO_VIEWS_OPTIMAM + EXTRA_VIEWS_OPTIMAM:
                continue
            if image.manufacturer in manufacturers:
                total_images += 1
                image_list.append(image.path)
                image_ids.append(image.id)
                for annotation in image.annotations:
                    if any(item in annotation.pathologies for item in pathologies):
                        total_annotations += 1
                        if image.view in CC_VIEWS_OPTIMAM:
                            view['CC'] += 1
                        elif image.view in MLO_VIEWS_OPTIMAM:
                            view['MLO'] += 1
                        mass_status[image.status] += 1
                        # breast_density[image.breast_density] += 1
                        if image.age < 50:
                            age['<50'] += 1
                        elif image.age >= 50 and image.age < 60:
                            age['50 to 60'] += 1
                        elif image.age >= 60 and image.age < 70:
                            age['60 to 70'] += 1
                        else:
                            age['>70'] += 1
                        site[image.site] += 1
    
    #print(f'{manufacturers}\n-----------')
    #print(f'Total clients: {total_clients} - Total images: {total_images} - total masses {total_annotations}')
    print(f'Images by Status: {mass_status}')
    # print(f'Images by Breast Density (ACR): {breast_density}')
    print(f'Images by View: {view}')
    print(f'Images by Age: {age}')
    print(f'Images by Site: {site}')

    # duplicates = set([x for x in image_list if image_list.count(x) > 1])
    # non_duplicates = list(set(image_list))
    # duplicates_ids = set([x for x in image_ids if image_ids.count(x) > 1])

if __name__ == '__main__':

    # Full scans
    # info_csv='/home/lidia/Datasets/OPTIMAM/png_screening/client_images_screening.csv'
    # dataset_path='/home/lidia/Datasets/OPTIMAM/png_screening/images'
    # output_path = '/home/lidia/Datasets/OPTIMAM/png_screening/detection'
    # cropped_to_breast = False
    # fit_to_breast = False

    # Cropped scans
    info_csv='/home/lidia/Datasets/OPTIMAM/png_screening_cropped_fixed/client_images_screening.csv'
    dataset_path='/home/lidia/Datasets/OPTIMAM/png_screening_cropped_fixed/images'
    output_path = '/home/lidia/Datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn'
    # info_csv='/home/lidia/Datasets/OPTIMAM/png_screening_cropped/client_images_screening.csv'
    # dataset_path='/home/lidia/Datasets/OPTIMAM/png_screening_cropped/images'
    # output_path = '/home/lidia/Datasets/OPTIMAM/png_screening_cropped/detection/mask_rcnn'
    cropped_to_breast = True
    fit_to_breast = True

    # Cropped scans GPU Server
    # info_csv='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/client_images_screening.csv'
    # dataset_path='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/images'
    # output_path = '/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn'
    # cropped_to_breast = True
    # fit_to_breast = True

    use_data_aug = False
    if use_data_aug:
        data_aug_path = '/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density' 
        data_aug_info = '/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/image_info_high_density_aug.csv'
    else:
        data_aug_path = None
        data_aug_info = None

    plot_OPTIMAM = False
    plot_studies = False
    compute_train_landmarks = False
    test_landmarks = False
    # landmarks_name = 'optimam_train_hologic_landmarks_all_no_asymmetries'
    # landmarks_name = 'optimam_train_hologic_landmarks_mass_calc'
    # landmarks_name = 'optimam_train_hologic_mass_data_aug_high_density'
    landmarks_name = 'optimam_train_hologic_mass_only_data_aug_high_density'

    save_COCO = False
    class_name = 'mass' #'mass_calc' #'all_pathologies' #'calc' #'malignant_pathology' # 'abnormality' # 'mass'
    save_name = 'mass' #'mass_data_aug_high_density' # 'mass_only_data_aug_high_density'
    pathologies = ['mass'] #['mass', 'calcifications', 'suspicious_calcifications', 'architectural_distortion'] #['mass'] # None # ['mass']
    #pathologies = None
    use_status = False
    category_id_dict = {'mass': 0} #{'mass': 0, 'calcification': 1, 'distortion': 2} # {'Malignant_pathology': 0} # {'Benign_mass': 0, 'Malignant_mass': 1} # {'mass': 0}, {'abnormality': 0}

    detection = False
    load_max = -1
    if not Path(output_path).exists():
        Path(output_path).mkdir(parents=True)
    
    image_lists = [['/home/lidia/Datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_mass_train.txt',
                    '/home/lidia/Datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_mass_val.txt'],   
                   '/home/lidia/Datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_mass_test.txt',
                   '/home/lidia/Datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_SIEMENS_mass_test.txt',
                   '/home/lidia/Datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_GE_MEDICAL_SYSTEMS_mass_test.txt',
                   '/home/lidia/Datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_Philips_Digital_Mammography_Sweden_AB_mass_test.txt']
    for image_list in image_lists:
        print(image_list)
        with Timer('dataset_distribution'):
            get_dataset_distribution(info_csv, dataset_path, detection, cropped_to_breast, pathologies,
                                    image_list=image_list)

    optimam = OPTIMAMDataset(info_csv, dataset_path, detection=detection, load_max=load_max, 
                            cropped_to_breast=cropped_to_breast)
    # for site in SITES:
    #     site_images = []
    #     status_dict = {'Normal': 0, 'Benign': 0, 'Malignant': 0}
    #     site_clients = optimam.get_images_by_site(site)
    #     for client in site_clients:
    #         for study in client:
    #             for image in study:
    #                 status_dict[image.status] += 1
    #                 site_images.append(image)

    #     normal = status_dict['Normal']
    #     benign = status_dict['Benign']
    #     malignant = status_dict['Malignant']
    #     print(f'{site}: {len(site_images)} images - Normal: {normal} - Benign: {benign} - Malignant: {malignant}')

    print(f'Total clients in loaded dataset: {len(optimam)}')
    print(f'Pahologies selected: {pathologies}')
    print('-----------------------------------')
    print(f'Clients: {optimam.total_clients(pathologies)} - Images: {optimam.total_images(pathologies)} - Annotations: {optimam.total_annotations(pathologies)}')

    # output_path = '/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/normal_abnormal'
    # pathologies = ['calcifications', 'suspicious_calcifications']
    # save_dataset_abnormal_classification(optimam, pathologies, 'HOLOGIC, Inc.', output_path, out_prefix='OPTIMAM_abnormal_calc')
    # pathologies = ['architectural_distortion']
    # save_dataset_abnormal_classification(optimam, pathologies, 'HOLOGIC, Inc.', output_path, out_prefix='OPTIMAM_abnormal_distortion')
    # pathologies = ['focal_asymmetry']
    # save_dataset_abnormal_classification(optimam, pathologies, 'HOLOGIC, Inc.', output_path, out_prefix='OPTIMAM_abnormal_assymetries')
    # pathologies = ['mass', 'architectural_distortion', 'calcifications', 'suspicious_calcifications']
    # save_dataset_abnormal_classification(optimam, pathologies, 'HOLOGIC, Inc.', output_path, out_prefix='OPTIMAM_abnormal_all_no_asymmetries')

    if pathologies is None:
        data_optimam = optimam
    else:
        clients_pathology = optimam.get_clients_by_pathology_and_status(pathologies)
        data_optimam = clients_pathology

    if plot_OPTIMAM:
        if plot_studies:
            for client in data_optimam:
                for study in client:
                    study.plot_study(print_annotations=True, fit_to_breast=fit_to_breast)
        else:
            # Plot dataset
            optimam.plot_dataset(print_annotations=True, fit_to_breast=fit_to_breast, max=load_max)

    # test_manufacturers = ['Philips Digital Mammography Sweden AB', 'GE MEDICAL SYSTEMS', 'Philips Medical Systems', 'SIEMENS']
    # test_manufacturers = ['HOLOGIC, Inc.']
    # for man_selected in test_manufacturers:
    #     X = []
    #     y = []
    #     for client in data_optimam:
    #         images = client.get_images_by_pathology(pathologies)
    #         for image in images:
    #             if image.manufacturer == man_selected:
    #                 X.append(client.id)
    #                 y.append(image.status)
    #                 break
        
    #     print(f'Dataset {man_selected}')
    #     print('------------------------')
    #     manufacturer_str = man_selected.replace(' ', '_')
    #     save_dataset_split_to_COCO(info_csv, dataset_path, cropped_to_breast, pathologies, man_selected, client_ids=X,
    #                                 output_path=output_path, txt_out=f'OPTIMAM_{manufacturer_str}_{save_name}_test.txt',
    #                                 json_out=f'OPTIMAM_{manufacturer_str}_{save_name}_test.json',
    #                                 category_id_dict=category_id_dict)

    # Train, val, test splits
    manufacturer = 'HOLOGIC, Inc.'
    X = []
    y = []
    for client in data_optimam:
        images = client.get_images_by_pathology(pathologies)
        for image in images:
            if image.manufacturer == manufacturer:
                X.append(client.id)
                y.append(image.status)
                break
    train_split = 0.7
    val_split = 0.1
    test_split = 0.2
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1-train_split, stratify=y, random_state=SEED, shuffle=True)
    relative_frac_test = test_split / (val_split + test_split)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=relative_frac_test, stratify=y_temp, random_state=SEED, shuffle=True)
    assert len(X) == len(X_train) + len(X_val) + len(X_test)

    #save_images_paths(clients_selected, os.path.join(output_path, txt_out))

    if compute_train_landmarks:
        optimam_train = OPTIMAMDataset(info_csv, dataset_path, detection=True, load_max=-1, 
                                cropped_to_breast=cropped_to_breast, client_ids=X_train)
        train_clients = optimam_train.get_clients_by_pathology_and_status(pathologies)
        image_paths = []
        for client in train_clients:
            selected_images = client.get_images_by_pathology(pathologies)
            for image in selected_images:
                #if image.site == 'stge':
                #image_paths.append(image.path)
                if use_data_aug:
                    image_paths.append(os.path.join(data_aug_path, image.id + '.png'))

        landmarks_path = os.path.join(output_path, f'{landmarks_name}.pth')
        landmarks = get_hist_stand_landmarks(image_paths)
        torch.save(landmarks, landmarks_path)

    if test_landmarks:
        optimam_test = OPTIMAMDataset(info_csv, dataset_path, detection=True, load_max=-1, 
                                cropped_to_breast=cropped_to_breast, client_ids=X_test)
        test_clients = optimam.get_clients_by_pathology_and_status(pathologies)
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
        json_out = f'OPTIMAM_{save_name}_train_fix.json'
        save_dataset_split_to_COCO(info_csv, dataset_path, cropped_to_breast, pathologies, manufacturer, client_ids=X_train,
                                    output_path=output_path, txt_out=f'OPTIMAM_{save_name}_train_fix.txt',
                                    json_out=json_out, category_id_dict=category_id_dict,
                                    data_aug_path=data_aug_path, data_aug_info=data_aug_info)
        print()
        print('Val Split')
        print('------------------------')
        json_out = f'OPTIMAM_{save_name}_val_fix.json'
        save_dataset_split_to_COCO(info_csv, dataset_path, cropped_to_breast, pathologies, manufacturer, client_ids=X_val,
                                    output_path=output_path, txt_out=f'OPTIMAM_{save_name}_val_fix.txt',
                                    json_out=json_out, category_id_dict=category_id_dict,
                                    data_aug_path=None, data_aug_info=None)
        print()
        print('Test Split')
        print('------------------------')
        json_out = f'OPTIMAM_{save_name}_test_fix.json'
        save_dataset_split_to_COCO(info_csv, dataset_path, cropped_to_breast, pathologies, manufacturer, client_ids=X_test,
                                    output_path=output_path, txt_out=f'OPTIMAM_{save_name}_test_fix.txt',
                                    json_out=json_out, category_id_dict=category_id_dict,
                                    data_aug_path=None, data_aug_info=None)
    
    # test_images = '/home/lidia/Datasets/OPTIMAM/png_screening_cropped/detection/OPTIMAM_mass_test.txt'
    # image_paths = open(test_images,'r').read().split('\n')
    # image_ids = []
    # for path in image_paths:
    #     image_id = os.path.splitext(os.path.basename(path))[0]
    #     image_ids.append(image_id)

    # optimam = OPTIMAMDataset(info_csv, dataset_path,
    #                 detection=True, load_max=-1, image_ids=image_ids, 
    #                 cropped_to_breast=True)

    # Generate COCO Datastet for Detection
    # Train: 80% Validation 10% Test 10%
    # train_clients = int(0.8*len(data_optimam)+0.5)
    # train = data_optimam[0:train_clients]
    # val_clients = train_clients+int(0.1*len(data_optimam)+0.5)
    # validation = data_optimam[train_clients:val_clients]
    # test = data_optimam[val_clients:]

    # mass_site_distribution = {'adde': {'Malignant' :0, 'Benign': 0}, 
    #                           'jarv': {'Malignant' :0, 'Benign': 0},
    #                           'stge': {'Malignant' :0, 'Benign': 0}}
    # manufacturers_sites = {'adde': [], 
    #                        'jarv': [],
    #                        'stge': []}
    # for client in data_optimam:
    #     images = client.get_images_by_pathology(pathologies)
    #     for image in images:
    #         total_masses = image.total_annotations(pathologies)
    #         mass_site_distribution[image.site][image.status] += total_masses
    #         manufacturers_sites[image.site].append(image.manufacturer)

    # Check images in HOLOGIC manufacurer
    # images_sites = {'adde': 0, 
    #                  'jarv': 0,
    #                  'stge': 0}
    # mass_site_distribution = {'adde': {'Malignant' :0, 'Benign': 0}, 
    #         'jarv': {'Malignant' :0, 'Benign': 0},
    #         'stge': {'Malignant' :0, 'Benign': 0}}
    # for client in data_optimam:
    #     images = client.get_images_by_pathology(pathologies)
    #     for image in images:
    #         if image.manufacturer == manufacturer:
    #             images_sites[image.site] += 1
    #             total_masses = image.total_annotations(pathologies)
    #             mass_site_distribution[image.site][image.status] += total_masses

    # Inspect sites
    # SITES = ['adde', 'jarv', 'stge']
    # for client in train:
    #     for study in client:
    #         for image in study:
    #             if image.site == 'stge':
    #                 input_img = np.array(Image.open(image.path).convert('RGB'))
    #                 plot_image_opencv_fit_window(input_img, title=f'{client.site}', 
    #                                             screen_resolution=(1920, 1080), wait_key=True)