import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

VIEW_DICT = {1: ['RIGHT', 'CC'],
             2: ['LEFT', 'CC'],
             3: ['RIGHT', 'MLO'],
             4: ['LEFT', 'MLO']}

IMAGE_TYPE_DICT = {0: ['RIGHT', 'CC'],
                   1: ['LEFT', 'CC'],
                   2: ['RIGHT', 'MLO'],
                   3: ['LEFT', 'MLO'],
                   4: ['PR', 'MLO'],
                   5: ['PL', 'MLO'],
                   6: ['RF', 'CC'],
                   7: ['LF', 'CC'],
                   8: ['RF', 'MLO'],
                   9: ['LF', 'MLO']}

class BCDRDataset():

    def __init__(self, info_file:str, dataset_path:str, outlines_csv=None, load_max=0, classes=None):

        self.path = dataset_path
        self.info_file_csv = info_file
        self.outlines_csv = outlines_csv
        df_outlines =  None
        if outlines_csv:
            df_outlines = BCDRSegmentationDataset(outlines_csv=outlines_csv, dataset_path=dataset_path)

        info_file_csv = pd.read_csv(info_file, delimiter=',')
        info_file_csv = info_file_csv.astype(object).replace(np.nan, '')
        # info_file_csv.head()
        self._case =  {'filename':[], 'scan':[], 'mask':[], 'pathology':[], 'classification':[],
         'patient_id':[], 'laterality':[], 'view':[], 'age':[], 'ACR':[], 'study_id':[], 'lesion_id':[]}
        counter = 0
        for index, case in info_file_csv.iterrows():
            if load_max != 0 and counter >= load_max:
                break
            image_filename = case['image_filename']
            image_filename = image_filename.replace(' ', '')
            self._case['filename'].append(image_filename)
            self._case['scan'].append(os.path.join(self.path, image_filename))
            self._case['patient_id'].append(case['patient_id'])
            self._case['study_id'].append(case['study_id'])
            self._case['age'].append(case['age'])
            image_view = int(case['image_type_id'])
            self._case['laterality'].append(IMAGE_TYPE_DICT[image_view][0])
            self._case['view'].append(IMAGE_TYPE_DICT[image_view][1])
            if isinstance(case['density'], str):
                density = case['density'].replace(' ', '')
                if density == 'NaN':
                    self._case['ACR'].append(0)
                else:
                    self._case['ACR'].append(int(density))
            else:
                self._case['ACR'].append(int(case['density']))
            if df_outlines:
                index = df_outlines.get_outlines_index_by_filename(image_filename)
                if index:
                    patient_info = df_outlines[index]
                    self._case['classification'].append(patient_info['classification'])
                    self._case['lesion_id'].append(patient_info['lesion_id'])
                    self._case['pathology'].append(patient_info['pathology'])
                    self._case['mask'].append(patient_info['mask'])
                else:
                    self._case['classification'].append('Normal')
                    self._case['lesion_id'].append(None)
                    self._case['pathology'].append(None)
                    self._case['mask'].append(None)
            else:
                self._case['classification'].append('Normal')
                self._case['lesion_id'].append(None)
                self._case['pathology'].append(None)
                self._case['mask'].append(None)
            counter += 1

    def __len__(self):
        return len(self._case['scan'])

    def __getitem__(self, idx):
        d = {}
        for k, vs in self._case.items():
            value = vs[idx]
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
    
    def get_scan_number_by_classification(self, classification):
        if classification not in self._case['classification']:
            return None
        else:
            count = self._case['classification'].count(classification)
            return count
    
    def get_scan_number_by_density(self, ACR):
        if ACR not in self._case['ACR']:
            return None
        else:
            count = self._case['ACR'].count(ACR)
            return count
    
    def print_distribution(self):
        for ACR in [1, 2, 3, 4]:
            count = self._case['ACR'].count(ACR)
            index_list = [i for i, value in enumerate(self._case['ACR']) if value == ACR]
            normal = 0
            benign = 0
            malign = 0
            for idx in index_list:
                case = self.__getitem__(idx)
                if case['classification'] == 'Benign':
                    benign +=1
                elif case['classification'] == 'Malign':
                    malign +=1
                else:
                    normal +=1
            print(f'ACR: {ACR} - Total scans: {count}')
            print(f'Normal scans: {normal} - Benign scans: {benign} - Malignant scans: {malign}')



class BCDRSegmentationDataset():

    def __init__(self, outlines_csv:str, dataset_path:str, load_max=0, classes=None):

        self.path = dataset_path
        self.outlines_csv = outlines_csv
        outlines = pd.read_csv(outlines_csv, delimiter=',')
        outlines = outlines.astype(object).replace(np.nan, '')
        # outlines.head()
        self._case =  {'filename':[], 'scan':[], 'mask':[], 'pathology':[], 'classification':[],
         'patient_id':[], 'laterality':[], 'view':[], 'age':[], 'ACR':[], 'study_id':[], 'lesion_id':[]}
        counter = 0
        for index, case in outlines.iterrows():
            if load_max != 0 and counter >= load_max:
                break
            image_filename = case['image_filename']
            image_filename = image_filename.replace(' ', '')
            self._case['filename'].append(image_filename)
            self._case['scan'].append(os.path.join(self.path, image_filename))
            self._case['classification'].append(case['classification'].replace(' ', ''))
            self._case['age'].append(case['age'])
            self._case['patient_id'].append(case['patient_id'])
            self._case['study_id'].append(case['study_id'])
            self._case['lesion_id'].append(case['lesion_id'])
            if isinstance(case['density'], str):
                density = case['density'].replace(' ', '')
                if density == 'NaN':
                    self._case['ACR'].append(0)
                else:
                    self._case['ACR'].append(int(density))
            else:
                self._case['ACR'].append(int(case['density']))
            image_view = int(case['image_view'])
            if image_view > 4:
                self._case['laterality'].append('UNKNOWN')
                self._case['view'].append('UNKNOWN')
            else:
                self._case['laterality'].append(VIEW_DICT[image_view][0])
                self._case['view'].append(VIEW_DICT[image_view][1])
            self._case['mask'].append(os.path.join(self.path, image_filename[:-4] + '_mask_id_' + str(case['lesion_id']) + '.tif'))
            pathologies = []
            if int(case['mammography_nodule']):
                pathologies.append('nodule')
            if int(case['mammography_calcification']):
                pathologies.append('calcification')
            if int(case['mammography_microcalcification']):
                pathologies.append('microcalcification')
            if int(case['mammography_axillary_adenopathy']):
                pathologies.append('axillary_adenopathy')
            if int(case['mammography_architectural_distortion']):
                pathologies.append('architectural_distortion')
            if int(case['mammography_stroma_distortion']):
                pathologies.append('stroma_distortion')
            self._case['pathology'].append(pathologies)
            counter += 1

    def __len__(self):
        return len(self._case['scan'])

    def __getitem__(self, idx):
        d = {}
        # for k, vs in self._case.items():
        #     value = vs[idx]
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
    
    def get_outlines_index_by_filename(self, filename):
        if filename not in self._case['filename']:
            return None
        else:
            index = self._case['filename'].index(filename)
            return index
    
    def get_scan_number_by_classification(self, classification):
        if classification not in self._case['classification']:
            return None
        else:
            count = self._case['classification'].count(classification)
            return count
    
    def get_scan_number_by_density(self, ACR):
        if ACR not in self._case['ACR']:
            return None
        else:
            count = self._case['ACR'].count(ACR)
            return count
    
    def print_distribution(self):
        for ACR in [1, 2, 3, 4]:
            count = self._case['ACR'].count(ACR)
            index_list = [i for i, value in enumerate(self._case['ACR']) if value == ACR]
            normal = 0
            benign = 0
            malign = 0
            for idx in index_list:
                case = self.__getitem__(idx)
                if case['classification'] == 'Benign':
                    benign +=1
                elif case['classification'] == 'Malign':
                    malign +=1
                else:
                    normal +=1
            print(f'ACR: {ACR} - Total scans: {count}')
            print(f'Normal scans: {normal} - Benign scans: {benign} - Malignant scans: {malign}')