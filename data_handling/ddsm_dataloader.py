import os
from SimpleITK.SimpleITK import Crop
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import nibabel as nib
import torchio as tio
import matplotlib.pyplot as plt
from scipy import stats
from skimage.transform import resize
from typing import Tuple, List, Union

# Dataloader for DDSM
# Takes: path to the files and path to the CSV info file
# Returns: a class Dataset object 

LABELS_DICT = {'BENIGN': 0,
               'BENIGN_WITHOUT_CALLBACK': 1,
               'MALIGNANT': 2}
class DDSMDataset():

    def __init__(self, info_file:Union[List[str], str], dataset_path='', load_max=0, classes=None):
        self.path = dataset_path
        counter = 0
        self._case =  {'case_id':[], 'scan':[], 'mask':[], 'cropped_roi':[], 'pathology':[], 'patient_id':[], 'laterality':[], 'view':[],
                        'type':[], 'mask_id':[], 'birad':[], 'subtlety':[], 'density':[]}
        self.info_files = []
        if isinstance(info_file, list):
            self.info_files = info_file
        else:
            self.info_files.append(info_file)
        
        for file in self.info_files:
            info = pd.read_csv(file)
            info.head()
            df_acr = info.groupby(['breast_density', 'pathology']).size().reset_index(name='counts')
            print(df_acr)
            # self._ACR_stats = {'ACR':[], 'benign':[], 'malignant':[]}
            # for ACR in [1, 2, 3, 4]:
            #     benign = df_acr.loc[(df_acr['breast_density'] == ACR) & (df_acr['pathology'] == 'BENIGN')].reset_index()['counts'][0]
            #     benign_without_call = df_acr.loc[(df_acr['breast_density'] == ACR) & (df_acr['pathology'] == 'BENIGN_WITHOUT_CALLBACK')].reset_index()['counts'][0]
            #     malignant = df_acr.loc[(df_acr['breast_density'] == ACR) & (df_acr['pathology'] == 'MALIGNANT')].reset_index()['counts'][0]
            #     self._ACR_stats['ACR'].append(ACR)
            #     self._ACR_stats['benign'].append(benign+benign_without_call)
            #     self._ACR_stats['malignant'].append(malignant)

            for index, case in tqdm(info.iterrows()):
                if load_max != 0 and counter >= load_max:
                    break
                if case['status'] == 'OK':
                    pathology = case['pathology']
                    if classes == 2 and pathology == 'BENIGN_WITHOUT_CALLBACK':
                            pathology = 'BENIGN'
                    self._case['scan'].append(os.path.join(self.path, case['image_file_path']))
                    self._case['mask'].append(os.path.join(self.path, case['ROI_mask_file_path']))
                    self._case['cropped_roi'].append(os.path.join(self.path, case['cropped_image_file_path']))
                    self._case['pathology'].append(pathology)
                    self._case['patient_id'].append(case['patient_id'])
                    self._case['laterality'].append(case['left_or_right_breast'])
                    self._case['view'].append(case['image_view'])
                    self._case['type'].append(case['abnormality_type'])
                    self._case['mask_id'].append(case['abnormality_id'])
                    self._case['birad'].append(case['assessment'])
                    self._case['subtlety'].append(case['subtlety'])
                    self._case['density'].append(int(case['breast_density']))
                    self._case['case_id'].append(case['patient_id'] + "_" + case['left_or_right_breast'] + "_" + case['image_view'] +
                                            "_" + case['abnormality_type'] + "_" + str(case['abnormality_id']))
                    counter += 1

    def split_train_val_csv(self, val_percentage=0.15, classes=2):
        out_train_df_list = []
        out_val_df_list = []
        for file in self.info_files:
            self.val_set_counter = {'ACR':[], 'benign':[], 'malignant':[]}
            for i in range(len(self._ACR_stats['ACR'])):
                self.val_set_counter['ACR'].append(self._ACR_stats['ACR'][i])
                self.val_set_counter['benign'].append(int(self._ACR_stats['benign'][i]*val_percentage))
                self.val_set_counter['malignant'].append(int(self._ACR_stats['malignant'][i]*val_percentage))
            drop_train_idx = []
            train_df = pd.read_csv(file, index_col=0)
            out_val_df = pd.DataFrame(columns=train_df.columns)
            for index, case in tqdm(train_df.iterrows()):
                if case['status'] == 'OK':
                    pathology = case['pathology']
                    density = int(case['breast_density'])
                    if classes == 2 and pathology == 'BENIGN_WITHOUT_CALLBACK':
                            pathology = 'BENIGN'
                    acr_idx = self.val_set_counter['ACR'].index(density)
                    if pathology == 'BENIGN':
                        if self.val_set_counter['benign'][acr_idx] > 0:
                            # Insert row
                            out_val_df.loc[len(out_val_df.index)] = train_df.iloc[index]
                            # Update val count
                            self.val_set_counter['benign'][acr_idx] = self.val_set_counter['benign'][acr_idx] - 1
                            # Drop index from train list
                            drop_train_idx.append(index)
                    else:
                        if self.val_set_counter['malignant'][acr_idx] > 0:
                            # Insert row
                            out_val_df.loc[len(out_val_df.index)] = train_df.iloc[index]
                            # Update val count
                            self.val_set_counter['malignant'][acr_idx] = self.val_set_counter['malignant'][acr_idx] - 1
                            # Drop index from train list
                            drop_train_idx.append(index)

            out_train_df = train_df.drop(train_df.index[drop_train_idx])
            out_train_df.rename(columns={'Unnamed: 0.1':'Unnamed: 0'}, inplace=True)
            out_val_df.rename(columns={'Unnamed: 0.1':'Unnamed: 0'}, inplace=True)
            out_train_df.reset_index(inplace=True, drop=True)
            out_train_df_list.append(out_train_df)
            out_val_df_list.append(out_val_df)

        return out_train_df_list, out_val_df_list

            
    def __len__(self):
        return len(self._case['scan'])

    def __getitem__(self, idx):
        d = {}
        [d.update({k: vs[idx]}) for k, vs in self._case.items()]
        return d

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
    
    def get_pathology(self, scan_path):
        if scan_path not in self._case['scan']:
            return None
        else:
            index = self._case['scan'].index(scan_path)
            return self._case['pathology'][index]
    
    def get_pathology_by_case_id(self, case_id):
        if case_id not in self._case['case_id']:
            return None
        else:
            index = self._case['case_id'].index(case_id)
            return self._case['pathology'][index]
    
    def get_label(self, scan_path):
        if scan_path not in self._case['scan']:
            return None
        else:
            index = self._case['scan'].index(scan_path)
            return LABELS_DICT[self._case['pathology'][index]]

    def get_mask(self, scan_path):
        if scan_path not in self._case['scan']:
            return None
        else:
            index = self._case['scan'].index(scan_path)
            return self._case['mask'][index]
    
    def get_type(self, scan_path):
        if scan_path not in self._case['scan']:
            return None
        else:
            index = self._case['scan'].index(scan_path)
            return self._case['type'][index]
    
    def get_pathology_labels_dict(self):
        return LABELS_DICT

class CropToMask(tio.transforms.Transform):

    def __init__(self, margin: int):
        self.margin = margin
        self.args_names = ('margin',)
        super(CropToMask, self).__init__()

    def apply_transform(self, subject) -> tio.Subject:

        mask = np.array(subject['mask'].data[0])
        maskx = np.any(mask, axis=1)
        masky = np.any(mask, axis=0)
        x1 = np.argmax(maskx)
        y1 = np.argmax(masky)
        x2 = len(maskx) - np.argmax(maskx[::-1])
        y2 = len(masky) - np.argmax(masky[::-1])
        width = int((x2 - x1 + 1) * (1 + self.margin/100))
        height = int((y2 - y1 + 1) * (1 + self.margin/100))

        tranform = tio.CropOrPad((width, height, 1), mask_name='mask')
        tranformed_subject = tranform(subject)
        return tranformed_subject

class Resize(tio.transforms.Transform):

    def __init__(self, size:Tuple[int], anti_aliasing:bool, channels=1):
        self.size = size
        self.anti_aliasing = anti_aliasing
        self.channels = channels
        self.args_names = ('anti_aliasing','size')
        super(Resize, self).__init__()

    def apply_transform(self, subject) -> tio.Subject:
        resized_img = resize(subject['img'].data[0], self.size, order=1, preserve_range=True, anti_aliasing=self.anti_aliasing)
        resized_mask = resize(subject['mask'].data[0], self.size, order=1, preserve_range=True, anti_aliasing=self.anti_aliasing)
        if self.channels == 1:
            subject['img'].data = torch.from_numpy(resized_img).unsqueeze(0)
        else:
            subject['img'].data = torch.from_numpy(np.stack((resized_img,)*self.channels, axis=-4)).unsqueeze(0)[0,...]
        subject['mask'].data = torch.from_numpy(resized_mask).unsqueeze(0)
        return subject


def get_preprocessed_ddsm_mass_image(img_path, mask_path, output_tag='', margin=30, size=(256, 256, 1)):
    subject = tio.Subject(
                img=tio.ScalarImage(img_path),
                mask=tio.LabelMap(mask_path),
                )
    mask_2 = subject['mask'].data
    transform = tio.Compose((
        tio.ZNormalization(masking_method='mask'),
        CropToMask(margin=margin),
        Resize(size=size, anti_aliasing=True, channels=3))
    )
    transformed = transform(subject)
    normalized_img_path = os.path.splitext(os.path.splitext(img_path)[0])[0] + output_tag + '.nii'
    transformed['img'].save(normalized_img_path, squeeze=False)
    normalized_mask_path = os.path.splitext(os.path.splitext(mask_path)[0])[0] + output_tag + '.nii'
    transformed['mask'].save(normalized_mask_path, squeeze=False)
    return normalized_img_path, normalized_mask_path