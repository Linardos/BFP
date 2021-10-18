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

from src.preprocessing.histogram_standardization import get_hist_stand_landmarks, apply_hist_stand_landmarks
from src.preprocessing.histogram_standardization import plot_img_and_hist

SEED = 999
np.random.seed(SEED)

# 723 female and 1 male
MLO_VIEW = ['MLO']
CC_VIEW = ['CC']
PATHOLOGIES = ['nodule', 'calcification', 'microcalcification', 'axillary_adenopathy',
                'architectural_distortion', 'stroma_distortion']
MANUFACTURERS = [''] #TODO
SITES = ['Centro Hospitalar São João at University of Porto (FMUP-HSJ)']
PIXEL_SIZE = [0.07, 0.07] #mm #TODO
RESOLUTION = 2e14 #14 bit resultion TIFFs
IMAGE_MATRICES = [(4084, 3328), (3328, 2560), (3072, 2816)] # height, width
STATUS = ['Normal', 'Benign', 'Malign'] #biopsy proven
ACR_DESCRIPTIONS = {1: 'Fatty', 
                    2: 'Scattered',
                    3: 'Heterogeneously dense',
                    4: 'Extremely dense'} # BI-RADS standard

class BCDRBBox():
    def __init__(self, x1, x2, y1, y2):
        self.x1 = int(x1)
        self.x2 = int(x2)
        self.y1 = int(y1)
        self.y2 = int(y2)
    
    def get_top_left_bottom_right(self):
        top, left = self.x1, self.y1
        bottom, right = self.x2, self.y2
        return top, left, bottom, right
    
    def get_xmin_xmax_ymin_ymax(self):
        xmin, ymin = self.x1, self.y1
        xmax, ymax = self.x2, self.y2
        return xmin, xmax, ymin, ymax

class BCDRAnnotation():
    def __init__(self, lesion_id, segmentation_id):
        if lesion_id == '':
            lesion_id = 0
        self.lesion_id = int(lesion_id)
        self.segmentation_id = int(segmentation_id)
        self.pathologies = []
        self.status = None
        self.bbox = None
        self.segmentation = None
    def set_status(self, status):
        if status == 'Malign':
            status = 'Malignant'
        self.status = status
    def set_pathologies(self, pathologies):
        for pathology in pathologies:
            if pathology in PATHOLOGIES:
                if pathology == 'nodule':
                    pathology = 'mass'
                self.pathologies.append(pathology)
    def set_bbox(self, bbox:BCDRBBox):
        self.bbox = bbox
    def set_segmentation(self, lw_x_points, lw_y_points):
        # Read segmentation
        lw_x_points = list(map(int, lw_x_points.split(" ")[1:]))
        lw_y_points = list(map(int, lw_y_points.split(" ")[1:]))
        outline_lesion = []
        for i in range(len(lw_x_points)):
            outline_lesion.append([lw_x_points[i], lw_y_points[i]])
        # Close polygon
        outline_lesion.append([lw_x_points[0], lw_y_points[0]])
        self.segmentation = np.array(outline_lesion)
    
    def get_top_left_bottom_right(self):
        return self.bbox.get_top_left_bottom_right()
    
    def get_xmin_xmax_ymin_ymax(self):
        return self.bbox.get_xmin_xmax_ymin_ymax()

class BCDRImage():
    def __init__(self, scan_path):
        self.id = None
        self.path=scan_path
        self.status = None
        self.site = None
        self.manufacturer = None
        self.view = None
        self.laterality = None
        self.implant = None
        self.pixel_spacing = None
        self.age = None
        self.annotations = []
        self.width = None
        self.height = None
        self.cropped_scan = False
        self.breast_width = None
        self.breast_height = None
        self.breast_xmin = None
        self.breast_xmax = None
        self.breast_ymin = None
        self.breast_ymax = None
        self.breast_density = None
    # Set methods
    def set_image_id(self, image_id):
        self.id = image_id
    def set_status(self, status):
        if status == 'Malign':
            status = 'Malignant'
        self.status = status
    def set_age(self, age):
        self.age = age
    def set_site(self, site):
        self.site = site
    def set_breast_density(self, breast_density):
        self.breast_density = int(breast_density)
    def set_manufacturer(self, manufacturer):
        self.manufacturer = manufacturer
    def set_view(self, view):
        if view in CC_VIEW:
            self.view = view
            return True
        elif view in MLO_VIEW:
            self.view = view
            return True
        else:
            print(f'Error: view {view} not found in list -> Discard image')
            return False
    def set_laterality(self, laterality):
        if laterality in ['RIGHT', 'R']:
            self.laterality = 'R'
        else:
            self.laterality = 'L'
    def set_implant(self, implant):
        if implant == 'YES':
            self.implant = True
        else:
            self.implant = False
    def set_pixel_spacing(self, pixel_spacing):
        self.pixel_spacing = pixel_spacing
    def set_breast_region(self, xmin, xmax, ymin, ymax):
        self.breast_xmin = int(xmin)
        self.breast_xmax = int(xmax)
        self.breast_ymin = int(ymin)
        self.breast_ymax = int(ymax)
    def set_cropped_scan(self, cropped_scan):
        self.cropped_scan = cropped_scan
    def add_annotation(self, annotation:BCDRAnnotation):
        self.annotations.append(annotation)

    def open_pil(self, print_annotations=False, color_bbox=(0, 255, 0),
                bbox_thickness=4, fit_to_breast=False):
        img_pil = Image.open(self.path).convert('RGB')
        img_pil = np.array(img_pil)
        if fit_to_breast and not self.cropped_scan:
            img_pil, coord = self.fit_to_breast(channels=3)
        if print_annotations and self.annotations: 
            for annotation in self.annotations:
                top, left, bottom, right = annotation.get_top_left_bottom_right()
                cv2.rectangle(img_pil, (top, left), (bottom, right), color_bbox, bbox_thickness)
        return img_pil
    
    def total_annotations(self, pathologies=None):
        if pathologies:
            counter = 0
            for annotation in self.annotations:
                if any(item in annotation.pathologies for item in pathologies):
                    counter += 1
            return counter
        else:
            return len(self.annotations)
    
    def get_pathologies(self):
        pathologies = []
        for annotation in self.annotations:
            for pathology in annotation.pathologies:
                pathologies.append(pathology)
        return list(set(pathologies))

    def fit_to_breast(self, channels=1):
        low_int_threshold = 0.05
        max_value = 255
        img_pil = Image.open(self.path)
        img_pil = np.array(img_pil)
        height, width = img_pil.shape[0], img_pil.shape[1]
        img_8u = (img_pil.astype('float32')/img_pil.max()*max_value).astype('uint8')
        
        low_th = int(max_value*low_int_threshold)
        _, img_bin = cv2.threshold(img_8u, low_th, maxval=max_value, type=cv2.THRESH_BINARY)
        contours,_ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cont_areas = [ cv2.contourArea(cont) for cont in contours ]
        idx = np.argmax(cont_areas)
        # # fill the contour.
        # breast_mask = cv2.drawContours(np.zeros_like(img_bin), contours, idx, 255, -1) 
        # # segment the breast.
        # img_breast_only = cv2.bitwise_and(img_pil, img_pil, mask=breast_mask)
        x,y,w,h = cv2.boundingRect(contours[idx])
        if channels == 3:
            img_pil = Image.open(self.path).convert('RGB')
            img_pil = np.array(img_pil)
        img_breast = img_pil[y:y+h, x:x+w]
        (xmin, xmax, ymin, ymax) = (x, x + w, y, y + h)
        self.breast_xmin = xmin
        self.breast_xmax = xmax
        self.breast_ymin = ymin
        self.breast_ymax = ymax
        for annotation in self.annotations:
            annotation.set_breast_xmin_xmax_ymin_ymax(xmin, xmax, ymin, ymax)
        return img_breast, (xmin, xmax, ymin, ymax)

    def generate_COCO_dict(self, image_id, obj_count, pathologies=None, fit_to_breast=False,
                            category_id_dict={'Benign_mass': 0, 'Malignant_mass': 1},
                            use_status=False):
        if not self.height:
            self.height, self.width = np.array(Image.open(self.path)).shape[:2]
        annotations_elem = []
        img_elem = dict(
                file_name = self.path,
                height=self.height,
                width=self.width,
                id=image_id)
        for annotation in self.annotations:
            if pathologies is None:
                add_annotation = True
            elif any(item in annotation.pathologies for item in pathologies):
                add_annotation = True
            else:
                add_annotation = False
            if add_annotation:
                if use_status:
                    if pathologies is not None:
                        if annotation.status + '_' + pathologies[0] in category_id_dict.keys():
                            category_id = category_id_dict[annotation.status + '_' + pathologies[0]]
                        else:
                            break
                    else:
                        if annotation.status + '_pathology' in category_id_dict.keys():
                            category_id = category_id_dict[annotation.status + '_pathology']
                        else:
                            break
                else:
                    if 'mass' in annotation.pathologies and 'mass' in category_id_dict.keys():
                        pathology_selected = 'mass'
                    elif ('calcifications' in annotation.pathologies or 'suspicious_calcifications' in annotation.pathologies) and 'calcification' in category_id_dict.keys():
                        pathology_selected = 'calcification'
                    elif 'architectural_distortion' in annotation.pathologies and 'distortion' in category_id_dict.keys():
                        pathology_selected = 'distortion'
                    else:
                        continue
                    category_id = category_id_dict[pathology_selected]
                xmin, xmax, ymin, ymax = annotation.get_xmin_xmax_ymin_ymax(fit_to_breast)
                #poly = [xmin, ymin],[xmax, ymin],[xmax, ymax],[xmin, ymax]
                annot_elem = dict(
                    image_id=image_id,
                    id=obj_count,
                    bbox=[xmin, ymin, xmax - xmin, ymax - ymin],
                    area=(xmax - xmin)*(ymax - ymin),
                    segmentation=[[xmin, ymin, int((xmax-xmin)/2) + xmin, ymin, \
                       xmax, ymin, xmax, int((ymax-ymin)/2) + ymin, \
                       xmax, ymax, int((xmax-xmin)/2) + xmin, ymax, \
                       xmin, ymax, xmin, int((ymax-ymin)/2) + ymin]],
                    category_id=category_id,
                    iscrowd=0
                    )
                annotations_elem.append(annot_elem)
                obj_count += 1 # each bbox increases the counter (here only one per image)
                if False:
                    #poly = [xmin, ymin],[xmax, ymin],[xmax, ymax],[xmin, ymax]
                    poly = [xmin, ymin], [int((xmax-xmin)/2) + xmin, ymin], \
                       [xmax, ymin], [xmax, int((ymax-ymin)/2) + ymin], \
                       [xmax, ymax], [int((xmax-xmin)/2) + xmin, ymax], \
                       [xmin, ymax], [xmin, int((ymax-ymin)/2) + ymin]
                    img = cv2.cvtColor(np.array(Image.open(self.path).convert('RGB')), cv2.COLOR_BGR2RGB)
                    cv2.polylines(img, [np.array(poly)], True, (0, 255, 0), 2)
                    plt.figure()
                    plt.imshow(img)
                    #plt.imsave('./test.png', img)
                    plt.show()
                    # img_pil = self.open_pil(print_annotations=True)
                    # plot_image_opencv_fit_window(img_pil, title='BCDR Scan', 
                    #                          screen_resolution=(1920, 1080), wait_key=True)
        return img_elem, annotations_elem, obj_count


class BCDRStudy():
    def __init__(self, study_id):
        self.id = study_id
        self.images = []
    
    def add_image(self, image:BCDRImage):
        self.images.append(image)

    def total_images(self, pathologies=None, status=None):
        if pathologies:
            counter = 0
            for image in self.images:
                image_pathologies = image.get_pathologies()
                if any(item in image_pathologies for item in pathologies):
                    if status:
                        if image.status in list(status):
                            counter += 1
                    else:
                        counter += 1
            return counter
        else:
            if status:
                counter = 0
                for image in self.images:
                    if image.status in list(status):
                        counter += 1
                return counter
            else:
                return len(self.images)
            
    def total_annotations(self, pathologies=None):
        counter = 0
        for image in self.images:
            counter += image.total_annotations(pathologies)
        return counter

    def get_image(self, image_id):
        for image in self.images:
            if image.id == image_id:
                return image
        return None
    
    def get_images_by_pathology(self, pathologies):
        if pathologies is None:
            return self.images
        images = []
        for image in self.images:
            image_pathologies = image.get_pathologies()
            if any(item in image_pathologies for item in pathologies):
                images.append(image)
        return images
    
    def get_images_by_status(self, status):
        images = []
        for image in self.images:
            if image.status in list(status):
                images.append(image)
        return images
    
    def get_images_by_view_laterality(self, view, laterality):
        images = []
        for image in self.images:
            if image.laterality == laterality:
                if image.view in CC_VIEW and view in CC_VIEW:
                    images.append(image)
                elif image.view in MLO_VIEW and view in MLO_VIEW:
                    images.append(image)
        return images
    
    def plot_study(self, print_annotations=True, fit_to_breast=False):
        title = ''
        multi_view = []
        img_height, img_width = None, None
        
        for view, laterality in zip(['CC', 'CC', 'MLO', 'MLO'], ['L', 'R', 'L', 'R']):
            view_img = self.get_images_by_view_laterality(view, laterality)
            if len(view_img):
                img_pil = view_img[0].open_pil(print_annotations=print_annotations,
                                fit_to_breast=fit_to_breast)
                if not img_width:
                    img_height, img_width = img_pil.shape[0], img_pil.shape[1]
                else:
                    img_pil = cv2.resize(img_pil, dsize=(img_width, img_height), interpolation=cv2.INTER_CUBIC)
                multi_view.append(img_pil)
                title += f' {laterality}{view} |'

        if len(multi_view) == 4:
            img_pil = np.concatenate((multi_view[0], multi_view[2], multi_view[1], multi_view[3]), axis=1)
        else:
            for i, image in enumerate(multi_view):
                if i+1 == len(multi_view): break
                img_pil = np.concatenate((multi_view[i], multi_view[i+1]), axis=1)
        print(title)
        if img_pil is not None:
            plot_image_opencv_fit_window(img_pil, title='BCDR Studies', 
                                        screen_resolution=(1920, 1080), wait_key=True)
            
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx]
    def __iter__(self):
        self.n = 0
        return self
    def __next__(self):
        if self.n < len(self):
            d = self.__getitem__(self.n)
            self.n += 1
            return d
        else:
            raise StopIteration
    

class BCDRClient():
    def __init__(self, client_id, breast_density):
        self.id=client_id
        self.breast_density=breast_density
        self.studies = []

    def update_study(self, study:BCDRStudy):
        for idx, s in enumerate(self.studies):
            if s.id == study.id:
                self.studies[idx] = study
                return
        self.studies.append(study)
    def total_studies(self, pathologies=None):
        if not pathologies:
            return len(self.studies)
        else:
            counter = 0
            for study in self.studies:
                image_counter = study.total_images(pathologies)
                if image_counter:
                    counter += 1
            return counter
    def total_images(self, pathologies=None, status=None):
        counter = 0
        for study in self.studies:
            counter += study.total_images(pathologies, status)
        return counter
    
    def total_annotations(self, pathologies=None):
        counter = 0
        for study in self.studies:
            counter += study.total_annotations(pathologies)
        return counter

    def get_study(self, study_id):
        for study in self.studies:
            if study.id == study_id:
                return study
        return None

    def get_images_by_pathology(self, pathologies):
        client_images = []
        for study in self.studies:
            images = study.get_images_by_pathology(pathologies)
            if len(images):
                for image in images:
                    client_images.append(image)
        return client_images
    
    def get_images_by_status(self, status):
        client_images = []
        for study in self.studies:
            images = study.get_images_by_status(status)
            if len(images):
                for image in images:
                    client_images.append(image)
        return client_images

    def get_image(self, image_id):
        for study in self.studies:
            image = study.get_image(image_id)
            if image:
                return image
        return image

    def __len__(self):
        return len(self.studies)
    def __getitem__(self, idx):
        return self.studies[idx]
    def __iter__(self):
        self.n = 0
        return self
    def __next__(self):
        if self.n < len(self):
            d = self.__getitem__(self.n)
            self.n += 1
            return d
        else:
            raise StopIteration

class BCDRDataset():

    def __init__(self, info_csv:str, dataset_path:str, detection=False, load_max=-1, 
                image_ids=None, cropped_scans=False, client_ids=None):
        if isinstance(info_csv, str):
            info_csv = [info_csv]
        if isinstance(dataset_path, str):
            dataset_path = [dataset_path]
        self.dataset_path = dataset_path
        self.info_csv = info_csv
        self.load_max = load_max
        self.clients = []
        self.images_ctr = 0
        self.annotation_ctr = 0
        self.cropped_scans = False
        # if detection:
        #     df_bbox = info.loc[info['Mass_0_x2'] != '']
        #     bbox_counts = df_bbox.groupby(['filename']).size().reset_index(name='counts')
        
        def get_client(client_id, breast_density):
            for client in self.clients:
                if client.id == client_id:
                    return client
            new_client = BCDRClient(client_id, breast_density)
            return new_client
    
        def update_client(client:BCDRClient):
            for idx, c in enumerate(self.clients):
                if c.id == client.id:
                    self.clients[idx] = client
                    return
            self.clients.append(client)

        for dataset_path_i, info_csv_i in zip(dataset_path, info_csv):
            dataset_name = os.path.basename(dataset_path_i)
            info = pd.read_csv(info_csv_i)
            info = info.astype(object).replace(np.nan, '')
            if detection:
                df_bbox = info.loc[info['lesion_x2'] != '']
                unique_patient_id_df = df_bbox.groupby(['patient_id'], as_index=False)
            else:
                unique_patient_id_df = info.groupby(['patient_id'], as_index=False)
            for patient_id, patient_group in unique_patient_id_df:
                if client_ids:
                    if patient_group.patient_id.values[0] not in client_ids:
                        continue

                unique_study_id_df = patient_group.groupby(['study_id'], as_index=False)
                for study_id, patient_study_group in unique_study_id_df:
                    unique_series_df = patient_group.groupby(['series'], as_index=False)
                    for series_id, series_group in unique_series_df:
                        unique_image_id_df = series_group.groupby(['scan_path'], as_index=False)
                        client_name = dataset_name + '_' + str(patient_id)
                        row_client = get_client(client_name, patient_group.density.values[0])
                        bool_update_client = False
                        for image_name, image_group in unique_image_id_df:
                            if image_ids:
                                image_name = image_name.replace(dataset_path_i + '/', '')
                                if image_name not in image_ids:
                                    break
                            # Create new image
                            scan_png_path = image_group.scan_path.values[0]
                            scan_path = os.path.join(dataset_path_i, scan_png_path)
                            # Create new BCDRImage
                            new_image = BCDRImage(scan_path)
                            valid_view = new_image.set_view(image_group.view.values[0])
                            if not valid_view:
                                break
                            init_image = True
                            for idx_mark, image in enumerate(image_group.itertuples()):
                                if detection and image.lesion_x2 == '': # Check if Annotation is available
                                    break
                                if init_image:
                                    new_image.set_image_id(image.scan_path)
                                    new_image.set_laterality(image.laterality)
                                    new_image.set_status(image.classification)
                                    new_image.set_site(SITES[0])
                                    new_image.set_manufacturer(MANUFACTURERS[0])
                                    new_image.set_pixel_spacing(PIXEL_SIZE[0])
                                    if client_name in ['BCDR-D01_dataset_511', 'BCDR-D01_dataset_129']:
                                        new_image.set_implant('YES')
                                    else:
                                        new_image.set_implant('NO')
                                    new_image.set_breast_region(image.breast_x1, image.breast_x2, image.breast_y1, image.breast_y2)
                                    new_image.set_cropped_scan(cropped_scans)
                                    new_image.width = image.scan_width
                                    new_image.height = image.scan_height
                                    new_image.breast_width = image.breast_width
                                    new_image.breast_height = image.breast_height
                                    new_image.set_breast_density(image.density)
                                    new_image.set_age(image.age)
                                    init_image = False
                                if image.lesion_x2 != '':
                                    new_annotation = BCDRAnnotation(image.lesion_id, image.segmentation_id)
                                    new_annotation.set_status(image.classification)
                                    new_annotation.set_pathologies(image.lesion_pathologies.strip('][').replace("'", '').split(', '))
                                    new_annotation.set_bbox(BCDRBBox(image.lesion_x1, image.lesion_x2,
                                                                        image.lesion_y1, image.lesion_y2))
                                    new_image.add_annotation(new_annotation)
                                    self.annotation_ctr += 1
  
                            if not init_image:
                                study_name = str(study_id) + '_' + str(series_id)
                                # Update study
                                row_study = row_client.get_study(study_name)
                                if row_study is None:
                                    row_study = BCDRStudy(study_name)
                                row_study.add_image(new_image)
                                # Update client
                                row_client.update_study(row_study)
                                self.images_ctr += 1
                                bool_update_client = True
                                if self.images_ctr == self.load_max:
                                    break
                        if bool_update_client:
                            update_client(row_client)
                            if self.images_ctr == self.load_max:
                                break

    def total_clients(self, pathologies=None, status=None):
        if not pathologies and not status:
            return len(self.clients)
        else:
            counter = 0
            for idx, client in enumerate(self.clients):
                if client.total_images(pathologies, status) > 0:
                    counter += 1
            return counter

    def total_images(self, pathologies=None, status=None):
        counter = 0
        # TODO check if same self.images_ctr
        for client in self.clients:
            counter += client.total_images(pathologies, status)
        return counter

    def total_annotations(self, pathologies=None):
        # TODO check self.annotation_ctr
        counter = 0
        for client in self.clients:
            counter += client.total_annotations(pathologies)
        return counter
    
    def get_client(self, client_id):
        for client in self.clients:
            if client.id == client_id:
                return client
        return None

    def get_clients_by_pathology_and_status(self, pathologies, status=None):
        clients = []
        for client in self.clients:
            images = client.total_images(pathologies, status)
            if images > 0:
                clients.append(client)
        return clients
    
    def get_clients_by_status(self, status):
        clients = []
        for client in self.clients:
            images = client.total_images(status)
            if images > 0:
                clients.append(client)
        return clients

    def get_image(self, image_id):
        for client in self.clients:
            image = client.get_image(image_id)
            if image:
                return image
        return image

    def __len__(self):
        return len(self.clients)
    def __getitem__(self, idx):
        return self.clients[idx]
    def __iter__(self):
        self.n = 0
        return self
    def __next__(self):
        if self.n < len(self):
            d = self.__getitem__(self.n)
            self.n += 1
            return d
        else:
            raise StopIteration
    
    def plot_dataset(self, print_annotations=True, fit_to_breast=False, max=-1):
        image_ctr = 0
        for client in self.clients:
            for study in client:
                for image in study:
                    img_pil = image.open_pil(print_annotations=print_annotations,
                                fit_to_breast=fit_to_breast)
                    plot_image_opencv_fit_window(img_pil, title='BCDR Scan', 
                                             screen_resolution=(1920, 1080), wait_key=True)
                    image_ctr += 1
                    if image_ctr >= max:
                        return

def save_images_paths(dataset, txt_out_file):
    # Save image list in txt
    image_paths = []
    for client in dataset:
        if pathologies:
            images = client.get_images_by_pathology(pathologies)
            for image in images:
                image_paths.append(image.path)
        else:
            for study in client:
                for image in study:
                    image_paths.append(image.path)

    with open(txt_out_file, 'w') as f:
        f.write("\n".join(image_paths))


def save_dataset_split_to_COCO(info_csv, dataset_path, cropped_scans, pathologies, client_ids,
                                output_path, txt_out, json_out, category_id_dict):
        BCDR_clients = BCDRDataset(info_csv, dataset_path, detection=True, load_max=-1, 
                                cropped_scans=cropped_scans, client_ids=client_ids)
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
                                  use_status=False):
    annotations = []
    images_dict = []
    obj_count = 0
    image_id = 0
    if pathologies:
        for client in clients:
            images = client.get_images_by_pathology(pathologies)
            for image in images:
                img_elem, annot_elem, obj_count = image.generate_COCO_dict(image_id, obj_count, pathologies,
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
    info_csv= ['/home/lidia/Datasets/BCDR/cropped/BCDR-DN01_dataset/dataset_info.csv']
    dataset_path= ['/home/lidia/Datasets/BCDR/cropped/BCDR-DN01_dataset']
    output_path = '/home/lidia/Datasets/BCDR/style_transfer/1333_800/testB'
    # info_csv= ['/home/lidia/Datasets/BCDR/cropped/BCDR-D01_dataset/dataset_info.csv',
    #             '/home/lidia/Datasets/BCDR/cropped/BCDR-D02_dataset/dataset_info.csv']
    # dataset_path= ['/home/lidia/Datasets/BCDR/cropped/BCDR-D01_dataset',
    #                 '/home/lidia/Datasets/BCDR/cropped/BCDR-D02_dataset']
    # output_path = '/home/lidia/Datasets/BCDR/style_transfer/1333_800/testA'
    selected_breast_density = [4] #[0,1,2]
    out_height = 1333
    out_width = 800

    cropped_scans = True
    fit_to_breast = True

    plot_BCDR = False
    plot_studies = True
    # Compute Landmarks for Gistogram Standardisation
    compute_train_landmarks = False
    test_landmarks = False
    landmarks_name = 'BCDR_train_landmarks_mass'

    save_COCO = False

    # PATHOLOGIES = ['nodule', 'calcification', 'microcalcification', 'axillary_adenopathy',
    #             'architectural_distortion', 'stroma_distortion']
    class_name = 'mass'
    pathologies = None #['mass']
    use_status = False
    category_id_dict = {'mass': 0}

    detection = False
    load_max = -1
    if not Path(output_path).exists():
        Path(output_path).mkdir(parents=True)
    
    BCDR = BCDRDataset(info_csv, dataset_path, detection=detection, load_max=load_max, 
                            cropped_scans=cropped_scans)
    print(f'Total clients in loaded dataset: {len(BCDR)}')
    print(f'Pahologies selected: {pathologies}')
    print('-----------------------------------')
    print(f'Clients: {BCDR.total_clients(pathologies)} - Images: {BCDR.total_images(pathologies)} - Annotations: {BCDR.total_annotations(pathologies)}')

    data_BCDR = BCDR
    # if pathologies is None:
    #     data_BCDR = BCDR
    # else:
    #     clients_pathology = BCDR.get_clients_by_pathology_and_status(pathologies)
    #     data_BCDR = clients_pathology

    if plot_BCDR:
        if plot_studies:
            for client in data_BCDR:
                for study in client:
                    print(f'Client ID: {client.id}')
                    study.plot_study(print_annotations=True, fit_to_breast=fit_to_breast)
        else:
            # Plot dataset
            BCDR.plot_dataset(print_annotations=True, fit_to_breast=fit_to_breast, max=load_max)

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
                                cropped_scans=cropped_scans, client_ids=X_test)
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
        save_dataset_split_to_COCO(info_csv, dataset_path, cropped_scans, pathologies, client_ids=X_train,
                                    output_path=output_path, txt_out=f'BCDR_{class_name}_train.txt',
                                    json_out=f'BCDR_{class_name}_train.json', category_id_dict=category_id_dict)
        print()
        print('Val Split')
        print('------------------------')
        save_dataset_split_to_COCO(info_csv, dataset_path, cropped_scans, pathologies, client_ids=X_val,
                                    output_path=output_path, txt_out=f'BCDR_{class_name}_val.txt',
                                    json_out=f'BCDR_{class_name}_val.json', category_id_dict=category_id_dict)
        print()
        print('Test Split')
        print('------------------------')
        save_dataset_split_to_COCO(info_csv, dataset_path, cropped_scans, pathologies, client_ids=X_test,
                                    output_path=output_path, txt_out=f'BCDR_{class_name}_test.txt',
                                    json_out=f'BCDR_{class_name}_test.json', category_id_dict=category_id_dict)
