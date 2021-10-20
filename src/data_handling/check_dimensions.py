import os
import re
import sys
import uuid
import numpy as np
import pandas as pd
import nibabel as nib
import pickle
import pprint


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]


def get_dirs(path):
    dirs = os.listdir(path)
    dirs.sort(key=natural_keys)
    items = [f for f in dirs if os.path.isdir(os.path.join(path, f))]
    return items


def get_files(path, suffix=''):
    dirs = os.listdir(path)
    dirs.sort(key=natural_keys)
    items = [f for f in dirs if os.path.isfile(os.path.join(path, f)) and os.path.join(path, f).endswith(suffix)]
    return items


def check_dimensions(data_dir, info):
    '''Checks if the SCANs and MASKs have the same dimension
    Arguments
    info: pandas csv file with paths

    Output report csv:
    patient_id, lr_view, scan_shape, mask_shape, status
    P_000121, LEFT_CC, (3220, 3232, 1), (3220, 3232, 1), OK
    P_012322, RIGHT_MLO, (2121, 1434, 1), (3232, 1221, 1), FAIL
    '''
    report = list()
    for idx, record in info.iterrows():
        # print(record)
        scan_file = os.path.join(data_dir, record['image_file_path'])
        mask_file = os.path.join(data_dir, record['ROI_mask_file_path'])

        scan = nib.load(scan_file).get_fdata()
        mask = nib.load(mask_file).get_fdata()
        pid = record['patient_id']
        lr = record['left_or_right_breast']
        view = record['image_view']
        stat = scan.shape == mask.shape
        # print(pid, scan_file, mask_file)
        
        row = {'scan_shape': scan.shape,
               'mask_shape': mask.shape,
               'status': 'OK' if stat else 'FAIL'}
        report.append({**record, **row})
    df = pd.DataFrame(report)
    return df

if __name__ == "__main__":
    num_args = len(sys.argv)
    args = sys.argv
    if num_args != 3:
        print('TypeError: it takes exactly 2 arguments ({} given)'.format(num_args - 1))
        print(' '*5, 'Example usage: check_dimensions.py data_dir calc_test_description.csv')
        sys.exit(0)
    
    data_dir = args[1]
    info_file = args[2]
    info = pd.read_csv(info_file)
    report = check_dimensions(data_dir, info)
    report.to_csv('report_' + os.path.basename(info_file))
    print('Done')

