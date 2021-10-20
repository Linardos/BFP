import os
import sys
import numpy as np
import pandas as pd
import openpyxl
from pathlib import Path
import csv
import nibabel as nib
import yaml

from src.data_handling.inbreast_dataset import load_inbreast_mask

""" 
INbreast Structure
    UUID(short)
        LEFT_MLO
            scan.nii.gz
            mask_calc.nii.gz -> labels: 1, 2, ..., N -> There are N lesions
            mask_mass.nii.gz -> labels: 1, 2, ..., N -> There are N lesions
            (mask_pectoral.nii.gz -> Not saved yet)
        RIGHT_MLO
        LEFT_CC
        RIGHT_CC
"""

if __name__ == '__main__':

    dataset_folder = Path('/home/lidia/Datasets/InBreast')
    niftis_folder = Path(os.path.join(dataset_folder, 'AllPatients'))
    output_csv = os.path.join(dataset_folder, 'INbreast_updated.csv')

    dicoms_folder = Path(os.path.join(dataset_folder, 'AllDICOMs'))
    xml_folder = Path(os.path.join(dataset_folder, 'AllXML'))
    pectoral_folder = Path(os.path.join(dataset_folder, 'PectoralMuscle/Pectoral_Muscle_XML'))
    info_csv = os.path.join(dataset_folder, 'INbreast.csv')
    
    info = pd.read_csv(info_csv, delimiter=';', index_col=None)
    head = info.head()
    if not niftis_folder.exists():
        niftis_folder.mkdir(parents=True)
    # Convert DICOMs to NIFTI and rename files
    for path in Path(dicoms_folder).glob('**/*.dcm'):
        orig_img = str(path)
        img_splits = os.path.basename(orig_img).split('_')
        file_name = img_splits[0]
        patient_id = img_splits[1]
        patient = info.loc[info['File Name'] == int(file_name)]
        if patient.Laterality.values[0] == 'R':
            laterality = 'RIGHT'
        else:
            laterality = 'LEFT'
        view = patient.View.values[0]
        subfolder = laterality + '_' + view
        save_path = Path(os.path.join(os.path.join(niftis_folder, patient_id), subfolder))
        if not save_path.exists():
            save_path.mkdir(parents=True)

        # Check if scan file exists
        scans = 0
        if Path(os.path.join(str(save_path), 'scan_0.nii.gz')).exists():
            images_list = os.listdir(save_path)
            scans = sum([file.count("scan_") for file in images_list])
        scan_filename = 'scan_' + str(scans)

        row_idx = int(info.loc[info['File Name'] == int(file_name)].index.values)
        # Modify CSV info to add Patient ID and Scan path
        info.at[row_idx, 'Patient ID'] = patient_id
        info.at[row_idx, 'Scan'] = os.path.join(os.path.join(patient_id, subfolder), scan_filename + '.nii.gz')
        # Convert scan image to Nifti
        dcim2nii_cmd = "dcm2niix -s y -b n -f " + scan_filename + " -o " + str(save_path) + " -z y " + str(orig_img)
        os.system(dcim2nii_cmd)

        mask_type = 'Mass'
        img = nib.load(os.path.join(save_path, scan_filename + '.nii.gz'))
        data = img.get_fdata()
        for mask_type in ['Mass', 'Calcification']:
            if mask_type == 'Pectoral Muscle':
                xml = os.path.join(pectoral_folder, file_name + '_muscle.xml')
            else:
                xml = os.path.join(xml_folder, file_name + '.xml')
            if mask_type == 'Mass':
                roi, rois = load_inbreast_mask(xml, mask_type=mask_type, imshape=data.shape, split_mask=True)
            else:
                roi, _ = load_inbreast_mask(xml, mask_type=mask_type, imshape=data.shape)
            if roi is not None:
                roi = np.fliplr(roi)
                print(data.shape, roi.shape)
                roi_img = nib.Nifti1Image(roi, img.affine, nib.Nifti1Header())
                if mask_type == 'Pectoral Muscle':
                    nib.save(roi_img, os.path.join(save_path, 's' + str(scans) + '_mask_pectoral.nii.gz'))
                    info.at[row_idx, 'Mass'] = os.path.join(os.path.join(patient_id, subfolder), 's' + str(scans) + '_mask_pectoral.nii.gz')
                elif mask_type == 'Mass':
                    nib.save(roi_img, os.path.join(save_path, 's' + str(scans) + '_mask_mass.nii.gz'))
                    info.at[row_idx, 'Mass'] = os.path.join(os.path.join(patient_id, subfolder), 's' + str(scans) + '_mask_mass.nii.gz')
                    for idx, mass in enumerate(rois):
                        mass = np.fliplr(mass)
                        mass_img = nib.Nifti1Image(mass, img.affine, nib.Nifti1Header())
                        nib.save(mass_img, os.path.join(save_path, 's' + str(scans) + '_mask_mass_' + str(idx) + '.nii.gz'))
                        info.at[row_idx, 'Mass_' + str(idx)] = os.path.join(os.path.join(patient_id, subfolder), 's' + str(scans) + '_mask_mass_' + str(idx) + '.nii.gz')
                else:
                    nib.save(roi_img, os.path.join(save_path, 's' + str(scans) + '_mask_calc.nii.gz'))
                    info.at[row_idx, 'Calcification'] = os.path.join(os.path.join(patient_id, subfolder), 's' + str(scans) + '_mask_calc.nii.gz')

    info.drop(['Patient age'], axis=1, inplace=True)
    info.to_csv(output_csv, index=False)
