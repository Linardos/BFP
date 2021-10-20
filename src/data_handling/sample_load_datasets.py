import os
import sys
import numpy as np
from pathlib import Path
import yaml
import pandas as pd
from shutil import copyfile
from tqdm import tqdm
from src.data_handling.ddsm_dataloader import DDSMDataset
from src.data_handling.inbreast_dataset import INbreastDataset
from src.data_handling.bcdr_dataset import BCDRSegmentationDataset, BCDRDataset

if __name__ == '__main__':

    if len(sys.argv) > 1:
        config_file = Path(sys.argv[1])
    else:
        config_file = Path('/home/lidia/source/BreastCancer/src/configs/config_datasets.yaml')
    with open(config_file) as file:
        config = yaml.safe_load(file)

    dataset = config['dataset']
    breast_density = config['images']['breast_density']
    dataset_path = config['paths']['dataset']
    if 'info_csv' in config['paths']:
        info_csv = config['paths']['info_csv']
    if 'info_csv_test' in config['paths']:
        info_csv_test = config['paths']['info_csv_test']
        info_csv_train = config['paths']['info_csv_train']
    if 'outlines_csv' in config['paths']:
        outlines_csv = config['paths']['outlines_csv']
    else:
        outlines_csv = None
    if 'input_folder' in config['paths']:
        input_folder = config['paths']['input_folder']
    else:
        input_folder = dataset_path
    
    if 'output_folder' in config['paths']:
        output_folder = config['paths']['output_folder']
        # Make output folders
        if not Path(output_folder).exists():
            Path(output_folder).mkdir(parents=True)
    
    img_list = []
    lab_list = []
    if dataset == 'BCDR':
        # Load all dataset data
        df_dataset = BCDRDataset(info_file=info_csv, outlines_csv=outlines_csv,
        dataset_path=dataset_path)
        print(f'{dataset} dataset TOTAL scans: {len(df_dataset)}')
        num_normal = df_dataset.get_scan_number_by_classification('Normal')
        num_benign = df_dataset.get_scan_number_by_classification('Benign')
        num_malign = df_dataset.get_scan_number_by_classification('Malign')
        print(f'Normal scans: {num_normal} - Benign scans: {num_benign} - Malign scans: {num_malign}')
        acr_1 = df_dataset.get_scan_number_by_density(1)
        acr_2 = df_dataset.get_scan_number_by_density(2)
        acr_3 = df_dataset.get_scan_number_by_density(3)
        acr_4 = df_dataset.get_scan_number_by_density(4)
        print(f'ACR 1: {acr_1} - ACR 2: {acr_2} - ACR 3: {acr_3} - ACR 4: {acr_4}')
        print()
        df_dataset.print_distribution()

        # Load only mages with segmentation masks
        df_seg_dataset = BCDRSegmentationDataset(outlines_csv=outlines_csv, dataset_path=dataset_path)
        print(f'{dataset} dataset scans with segmentation: {len(df_seg_dataset)}')
        df_seg_dataset.print_distribution()
        
        with tqdm(total=len(df_seg_dataset)) as pbar:
            # Iterate trought the dictionary
            for case in df_seg_dataset:
                scan_img_path = case['scan']
                mask_img_path = case['mask']
                pathologies = case['pathology']
                classification = case['classification']
                if case['ACR'] in breast_density:
                    if classification == 'Benign':
                        label = 0
                    elif classification == 'Malign':
                        label = 1
                    img_list.append(scan_img_path)
                    lab_list.append(label) 
                    pbar.update(1)

    elif dataset == 'CBIS-DDSM':
        df_dataset = DDSMDataset(info_file=info_csv_test, dataset_path=dataset_path)
        # df_dataset_test = DDSMDataset(info_file=info_csv_test, dataset_path=dataset_path)
        # df_dataset_train = DDSMDataset(info_file=info_csv_train, dataset_path=dataset_path)
        print(dataset + ' dataset cases: ' + str(len(df_dataset)))
        with tqdm(total=len(df_dataset)) as pbar:
            # Iterate trought the dictionary
            for case in df_dataset:
                img_path = os.path.join(input_folder, case['case_id'] + '.png')
                pathology = case['pathology']
                if case['density'] in breast_density:
                    if pathology == 'BENIGN_WITHOUT_CALLBACK' or pathology == 'BENIGN':
                        label = 0
                    else:
                        label = 1
                    img_list.append(img_path)
                    lab_list.append(label) 
                    pbar.update(1)
                else:
                    print('Density out of accepted values:' + str(case['density']))
    elif dataset == 'INbreast':
        df_dataset = INbreastDataset(info_file=info_csv, dataset_path=dataset_path)
        print(dataset + ' dataset cases: ' + str(len(df_dataset)))
        with tqdm(total=len(df_dataset)) as pbar:
            for case in df_dataset:
                birad = case['birad']
                input = os.path.join(input_folder, birad + '/scans')
                img_path = os.path.join(input, case['case_id'] + '.png')
                # Skip missing ACR in 53582540 filename
                if case['ACR'] not in breast_density:
                    continue
                else:
                    if birad in ['1', '2', '3']:
                        label = 0 #Benign
                    else:
                        label = 1 #Malignant
                    img_list.append(img_path)
                    lab_list.append(label)
                    pbar.update(1)
    elif dataset == 'mini-MIAS':
        info = pd.read_csv(info_csv, delimiter=',', index_col=0)
        info = info.astype(object).replace(np.nan, '')
        with tqdm(total=info.shape[0]) as pbar:
            for case in info.iterrows():
                img_path = os.path.join(input_folder, case[0] + '.png')
                # F - Fatty 
                # G - Fatty-glandular
                # D - Dense-glandular
                if case[1]['Tissue'] in breast_density:
                    if case[1]['Abnormality'] == 'NORM':
                        # label = 0
                        continue
                    elif case[1]['Severity'] == 'M':
                        label = 1
                    elif case[1]['Severity'] == 'B':
                        label = 0
                    else:
                        continue
                    img_list.append(img_path)
                    lab_list.append(label)
                    pbar.update(1)
    else:
        print('Wrong dataset')

    print('Total images loaded: ' + str(len(img_list)))
