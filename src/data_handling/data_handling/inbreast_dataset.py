"""
Import an Osirix xml roi as a binary numpy array

Requires libraries:
nibabel
numpy
scikit-image
xml

Benjamin Irving
2014/03/31

Licence:
BSD

"""
from __future__ import print_function, division

import os
from skimage.draw import polygon
import numpy as np
import plistlib
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def load_inbreast_mask(mask_path, mask_type='Mass', imshape=(4084, 3328), split_mask=False):
    """
    This function loads a osirix xml region as a binary numpy array for INBREAST
    dataset

    @mask_path : Path to the xml file
    @mask_type : ROI type ['Mass' or 'Calcification']
    @imshape : The shape of the image as an array e.g. [4084, 3328]

    return: numpy array where positions in the roi are assigned a value of 1.

    """

    def load_point(point_string):
        x, y = tuple([float(num) for num in point_string.strip('()').split(',')])
        return y, x

    mask_rois = []
    mask_roi = None
    mask = np.zeros(imshape)
    empty = True
    if Path(mask_path).exists():
        with open(mask_path, 'rb') as mask_file:
            print(mask_path)
            plist_dict = plistlib.load(mask_file, fmt=plistlib.FMT_XML)['Images'][0]
            numRois = plist_dict['NumberOfROIs']
            rois = plist_dict['ROIs']
            assert len(rois) == numRois
            mask_value = 0
            for roi in rois:
                numPoints = roi['NumberOfPoints']
                points = roi['Point_px']
                name = roi['Name']
                if name != mask_type:
                    continue
                assert numPoints == len(points)  
                empty = False
                if split_mask:
                    mask_roi = np.zeros(imshape)
                mask_value = mask_value + 1
                points = [load_point(point) for point in points]
                if len(points) <= 2:
                    for point in points:
                        #mask[int(point[0]), int(point[1])] = mask_value
                        mask[int(point[1]), int(point[0])] = mask_value
                        if split_mask:
                            #mask_roi[int(point[0]), int(point[1])] = 1
                            mask_roi[int(point[1]), int(point[0])] = 1
                else:
                    x, y = zip(*points)
                    #x, y = np.array(x), np.array(y)
                    x, y = np.array(y), np.array(x)
                    poly_x, poly_y = polygon(x, y, shape=imshape)
                    mask[poly_x, poly_y] = mask_value
                    if split_mask:
                        mask_roi[poly_x, poly_y] = 1
                if split_mask:
                    mask_rois.append(mask_roi)
    if empty:
        return None, None
    else:
        return mask, mask_rois

class INbreastDataset():

    def __init__(self, info_file:str, dataset_path:str, load_max=0, classes=None):
            
        self.path = dataset_path
        self.info_file = info_file
        counter = 0
        info = pd.read_csv(info_file)
        info = info.astype(object).replace(np.nan, '')
        info.head()
        mass_cols = [col for col in info.columns if 'Mass_' in col]
        self._case =  {'case_id':[], 'filename':[], 'scan':[], 'scan_id':[], 'masks':[], 'pathology':[], 'patient_id':[], 'laterality':[], 'view':[],
                        'birad':[], 'ACR':[]}
        df_acr = info.groupby(['ACR', 'Bi-Rads']).size().reset_index(name='counts')
        # print(df_acr)
        self._ACR_stats = {'ACR':[], 'num_scans':[], 'benign':[], 'malignant':[]}
        for ACR in [1, 2, 3, 4]:
            benign_count = 0
            malignant_count = 0
            sel_acr = df_acr.loc[df_acr['ACR'] == ACR]
            for idx, sel_acr_row in sel_acr.iterrows():
                birad = int(list(sel_acr_row['Bi-Rads'])[0])
                if birad > 3:
                    malignant_count += int(sel_acr_row['counts'])
                else:
                    benign_count += int(sel_acr_row['counts'])
            self._ACR_stats['ACR'].append(ACR)
            self._ACR_stats['num_scans'].append(benign_count+malignant_count)
            self._ACR_stats['benign'].append(benign_count)
            self._ACR_stats['malignant'].append(malignant_count)
        
        for index, case in tqdm(info.iterrows()):
            if load_max != 0 and counter >= load_max:
                break
            self._case['birad'].append(case['Bi-Rads'])
            scan_idx = int(case['Scan'].split('/').pop()[5:6])
            self._case['scan_id'].append(scan_idx)
            birad = int(list(case['Bi-Rads'])[0])
            if birad > 3:
                pathology = 'MALIGNANT'
            else:
                pathology = 'BENIGN'
            self._case['scan'].append(os.path.join(self.path, case['Scan']))
            self._case['filename'].append(case['File Name'])
            self._case['patient_id'].append(case['Patient ID'])
            self._case['laterality'].append(case['Laterality'])
            self._case['view'].append(case['View'])
            self._case['pathology'].append(pathology)
            self._case['ACR'].append(case['ACR'])
            self._case['case_id'].append(case['Patient ID'] + "_" + case['Laterality'] + "_" + case['View'] +
                                    "_scan_" + str(scan_idx))
            mask_list = []
            mass_roi = case['Mass']
            if mass_roi:
                mask = {}
                mask['mask'] = os.path.join(self.path, mass_roi)
                mask['mask_id'] = -1
                mask['type'] = 'masses'
                mask_list.append(mask)
            for mass_idx, mass_col in enumerate(mass_cols):
                mass_roi = case[mass_col]
                if mass_roi:
                    mask = {}
                    mask['mask'] = os.path.join(self.path, mass_roi)
                    mask['mask_id'] = mass_idx
                    mask['type'] = 'mass'
                    mask_list.append(mask)
            calc_roi = case['Calcification']
            if calc_roi:
                mask = {}
                mask['mask'] = os.path.join(self.path, calc_roi)
                mask['mask_id'] = -1
                mask['type'] = 'calcifications'
                mask_list.append(mask)
            self._case['masks'].append(mask_list)
            counter += 1

    def split_train_val_csv(self, val_percentage=0.10, test_percentage=0.20, classes=2):
        
        self.val_set_counter = {'ACR':[], 'benign':[], 'malignant':[]}
        self.test_set_counter = {'ACR':[], 'benign':[], 'malignant':[]}
        for i in range(len(self._ACR_stats['ACR'])):
            self.val_set_counter['ACR'].append(self._ACR_stats['ACR'][i])
            if val_percentage == 0:
                self.val_set_counter['benign'].append(0)
                self.val_set_counter['malignant'].append(0)
            else:
                self.val_set_counter['benign'].append(max(1, int(self._ACR_stats['benign'][i]*val_percentage)))
                self.val_set_counter['malignant'].append(max(1, int(self._ACR_stats['malignant'][i]*val_percentage)))
            self.test_set_counter['ACR'].append(self._ACR_stats['ACR'][i])
            self.test_set_counter['benign'].append(max(1, int(self._ACR_stats['benign'][i]*test_percentage)))
            self.test_set_counter['malignant'].append(max(1, int(self._ACR_stats['malignant'][i]*test_percentage)))
        
        drop_train_idx = []
        train_df = pd.read_csv(self.info_file)
        if val_percentage == 0:
            out_val_df = None
        else:
            out_val_df = pd.DataFrame(columns=train_df.columns)
        out_test_df = pd.DataFrame(columns=train_df.columns)
        for index, case in tqdm(train_df.iterrows()):
            birad = int(list(case['Bi-Rads'])[0])
            if birad > 3:
                pathology = 'MALIGNANT'
            else:
                pathology = 'BENIGN'
            density = int(case['ACR'])
            if density in [1, 2, 3, 4]:
                acr_idx = self.val_set_counter['ACR'].index(density)
                if pathology == 'BENIGN':
                    if self.val_set_counter['benign'][acr_idx] > 0:
                        # Insert row
                        out_val_df.loc[len(out_val_df.index)] = train_df.iloc[index]
                        # Update val count
                        self.val_set_counter['benign'][acr_idx] = self.val_set_counter['benign'][acr_idx] - 1
                        # Drop index from train list
                        drop_train_idx.append(index)
                    elif self.test_set_counter['benign'][acr_idx] > 0:
                        # Insert row
                        out_test_df.loc[len(out_test_df.index)] = train_df.iloc[index]
                        # Update val count
                        self.test_set_counter['benign'][acr_idx] = self.test_set_counter['benign'][acr_idx] - 1
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
                    elif self.test_set_counter['malignant'][acr_idx] > 0:
                        # Insert row
                        out_test_df.loc[len(out_test_df.index)] = train_df.iloc[index]
                        # Update val count
                        self.test_set_counter['malignant'][acr_idx] = self.test_set_counter['malignant'][acr_idx] - 1
                        # Drop index from train list
                        drop_train_idx.append(index)

        out_train_df = train_df.drop(train_df.index[drop_train_idx])
        out_train_df.reset_index(inplace=True, drop=True)

        return out_train_df, out_val_df, out_test_df

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

    def get_masks(self, scan_path):
        if scan_path not in self._case['scan']:
            return None
        else:
            index = self._case['scan'].index(scan_path)
            return self._case['masks'][index]
    
    def get_masses_by_case_id(self, case_id):
        if case_id not in self._case['case_id']:
            return None
        else:
            index = self._case['case_id'].index(case_id)
            masses = None
            for masks in self._case['masks'][index]:
                if masks['type'] == 'masses':
                    masses = masks['mask']
            return masses

    def get_calcifications_by_case_id(self, case_id):
        if case_id not in self._case['case_id']:
            return None
        else:
            index = self._case['case_id'].index(case_id)
            calcifications = None
            for masks in self._case['masks'][index]:
                if masks['type'] == 'calcifications':
                    calcifications = masks['mask']
            return calcifications