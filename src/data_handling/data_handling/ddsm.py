'''
@author: Kaisar Kushibar (kaisar.kushibar@udg.edu)
@date: 30/Jan/2019
@requirements: dcm2niix
'''
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


def fix_tmps(logs):
    '''
    ['tmp',
     '/home/kaisar/Datasets/TCIA/DDSM/Images/CBIS-DDSM/Train/Mass-Training_P_02092_LEFT_MLO_1/07-20-2016-DDSM-20213/1.000000-cropped images-58924/1-1.dcm',
     '/home/kaisar/Datasets/TCIA/DDSM/Images/CBIS-DDSM-NIFTI/Train/P_02092/LEFT_MLO/tmp_1',
     'a231bc85c74946db9f9f5ab7b91062d2.nii.gz'
    ], 
    ['tmp',
     '/home/kaisar/Datasets/TCIA/DDSM/Images/CBIS-DDSM/Train/Mass-Training_P_02092_LEFT_MLO_1/07-21-2016-DDSM-14412/1.000000-ROI mask images-30086/1-1.dcm',
     '/home/kaisar/Datasets/TCIA/DDSM/Images/CBIS-DDSM-NIFTI/Train/P_02092/LEFT_MLO/tmp_1',
     'd633b102f5c94032a69da6a56e9e13be.nii.gz'
    ]
    '''
    pairs = dict()
    for log in logs:
        if log[0] == 'tmp':
            input_file = log[1]
            out_folder = log[2]
            file_name = log[3]
            if out_folder in pairs:
                pairs[out_folder].append(os.path.join(out_folder, file_name))
            else:
                pairs[out_folder] = [input_file, os.path.join(out_folder, file_name)]

    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(pairs)

    prefixes = ['Mass-Training_', 'Mass-Test_', 'Calc-Training_', 'Calc-Test_']
    # move the files up and rename 
    for out in pairs:
        # assume pairs[out][0] is mask and pairs[out][1] is roi
        input_file = pairs[out][0]
        mask_file_path = pairs[out][1]
        roi_file_path = pairs[out][2]

        # mask_file_size = os.stat(mask_file_path).st_size
        # roi_file_size = os.stat(roi_file_path).st_size

        # file sizes change after converting nii. need to load and compare shapes
        mask_file_size = np.prod(nib.load(mask_file_path).get_fdata().shape)
        roi_file_size = np.prod(nib.load(roi_file_path).get_fdata().shape)
                
        # larger shape is MASK and smaller is ROI
        # so if mask_size < roi_size, then swap them 
        if mask_file_size < roi_file_size:
            tmp = mask_file_path
            mask_file_path = roi_file_path
            roi_file_path = tmp
        
        tmp_dir_path = os.path.dirname(mask_file_path)
        tar_dir_path = os.path.dirname(tmp_dir_path)
        
        # get the index from suffix _1, _2, etc.
        index = tmp_dir_path[-2:]
        dirname, basename = os.path.split(input_file)
        while not basename.startswith(tuple(prefixes)):
            dirname, basename = os.path.split(dirname)
        lesion_type = '_mass' if basename.startswith('Mass') else '_calc'
        tar_mask_path = os.path.join(tar_dir_path, 'mask'+lesion_type+index+'.nii.gz')
        tar_roi_path = os.path.join(tar_dir_path, 'roi'+lesion_type+index+'.nii.gz')

        # move files
        os.system(f'mv {mask_file_path} {tar_mask_path}')
        os.system(f'mv {roi_file_path} {tar_roi_path}')
        # remove tmp dir
        if len(os.listdir(tmp_dir_path)) == 0:
            os.system(f'rmdir {tmp_dir_path}')
        else:
            print('TMP FOLDER IS NOT EMPTY:', tmp_dir_path)
            sys.exit(-1)

def dcm2niix(input_file, output_folder, save_filename):
    ec_input_file = input_file.replace(" ", "\\ ")
    os.system(f'dcm2niix -9 -s y -z y -f {save_filename} -o {output_folder} {ec_input_file}')
    # remove json
    json_file = os.path.join(output_folder, save_filename+'.json')
    os.system(f'rm {json_file}')

'''
CBIS-DDSM-NIFTI
    Train
        P_0001
            LEFT_MLO
                scan.nii.gz
                mask_calc_1.nii.gz
                mask_mass_N.nii.gz
                roi_calc_1.nii.gz
                roi_mass_N.nii.gz
            RIGHT_MLO
            LEFT_CC
            RIGHT_CC
INbreast
    UUID(short)
        LEFT_MLO
            scan.nii.gz
            mask_calc.nii.gz -> labels: 1, 2, ..., N -> There are N lesions
            mask_mass.nii.gz -> labels: 1, 2, ..., N -> There are N lesions
        RIGHT_MLO
        LEFT_CC
        RIGHT_CC
'''


def convert_to_nifti(input_file, output_folder, save_filename, is_gt, logs):
    prefixes = ['Mass-Training_', 'Mass-Test_', 'Calc-Training_', 'Calc-Test_']
    if not is_gt:
        # Scan
        logs.append([save_filename, input_file, output_folder, save_filename+'.nii.gz'])
        dcm2niix(input_file, output_folder, save_filename)
    else: 
        # Roi or mask
        # get the index from suffix _1, _2, etc.
        dirname, basename = os.path.split(input_file)
        while not basename.startswith(tuple(prefixes)):
            dirname, basename = os.path.split(dirname)
        index = basename[-2:]
        lesion_type = '_mass' if basename.startswith('Mass') else '_calc'
        if save_filename == 'mask' or save_filename == 'roi':
            logs.append([save_filename, input_file, output_folder, save_filename+lesion_type+index+'.nii.gz'])
            dcm2niix(input_file, output_folder, save_filename+lesion_type+index)
        else: 
            out = os.path.join(output_folder, 'tmp'+lesion_type+index)
            if not os.path.exists(out):
                os.mkdir(out)
            logs.append(['tmp', input_file, out, save_filename+'.nii.gz'])
            dcm2niix(input_file, out, save_filename)


def walk(cur_in, cur_out, level, is_gt_folder, logs):
    '''Recursively walks through the folders and converts dicoms to nifti and
    outputs the in output destination
    cur_in: input folder
    cur_out: output folder
    level: level of recursion

    max_level for cur_out is: < 2 CBIS-DDSM/Test/Mass-Test_P_*/
    if cur_out has suffix _1 then
        remove the suffix
        convert the files into tmp1.nii.gz and tmp2.nii.gz
        load these two files, give the larger shaped file name mask.nii.gz and smaller one roi.nii.gz
    '''
        
    # print(level, cur_in, cur_out, is_gt_folder)
    max_level = 2
    prefixes = ['Mass-Training_', 'Mass-Test_', 'Calc-Training_', 'Calc-Test_']
    suffixes = tuple(['_'+str(i) for i in range(1, 10)])
    views = ['_LEFT_CC', '_LEFT_MLO', '_RIGHT_CC', '_RIGHT_MLO']
    files = get_files(cur_in)
    if len(files) == 0:
        for d in get_dirs(cur_in):
            is_gt = is_gt_folder
            if level < max_level:
                next_out = os.path.join(cur_out, d)
                for pref in prefixes:
                    if os.path.basename(next_out).startswith(pref):
                        next_out = os.path.join(os.path.dirname(next_out),
                                                os.path.basename(next_out).replace(pref, ''))
                        break
                if next_out.endswith(suffixes):
                    is_gt = True
                    next_out = next_out[:-2]
                for view in views:
                    if os.path.basename(next_out).endswith(view):
                        next_out = os.path.join(os.path.dirname(next_out),
                                                os.path.basename(next_out).replace(view, ''),
                                                view[1:])
                if not os.path.exists(next_out):
                    os.makedirs(next_out, exist_ok=True)
            else:
                next_out = cur_out
            next_in = os.path.join(cur_in, d)
            walk(next_in, next_out, level+1, is_gt, logs)
    elif len(files) == 1 and not is_gt_folder:
        # SCAN
        scan_path = os.path.join(cur_in, files[0])
        # print('SCAN', scan_path, cur_out)
        # convert to nifti and save in new path
        convert_to_nifti(scan_path, cur_out, 'scan', is_gt=False, logs=logs)
    elif len(files) == 2 and is_gt_folder:
        # ROI and MASK
        # assume files[0] is mask and files[1] is roi
        mask_path = os.path.join(cur_in, files[0])
        roi_path = os.path.join(cur_in, files[1])

        mask_size = os.stat(mask_path).st_size
        roi_size = os.stat(roi_path).st_size

        # larger size is MASK and smaller is ROI
        # so if mask_size < roi_size, then swap them 
        if mask_size < roi_size:
            tmp = mask_path
            mask_path = roi_path
            roi_path = tmp
        # convert to nifti and save
        # print('ROI and MASK', mask_path, roi_path, cur_out)
        convert_to_nifti(mask_path, cur_out, 'mask', is_gt=True, logs=logs)
        convert_to_nifti(roi_path, cur_out, 'roi', is_gt=True, logs=logs)
    elif len(files) == 1 and is_gt_folder:
        # ROI or MASK
        # since we cannot compare them, save with a unique filename and run
        # a different script to assign a proper name
        roi_or_mask_path = os.path.join(cur_in, files[0])
        # convert to nifti and save in new path
        # print('ROI or MASK', roi_or_mask_path, cur_out)
        out_name = str(uuid.uuid4().hex)
        convert_to_nifti(roi_or_mask_path, cur_out, out_name, is_gt=True, logs=logs)
    else:
        logs.append(['fail', cur_in, cur_out])
        print('Something went wrong')
        print(len(files), files, is_gt_folder)
        print(cur_in, cur_out)


if __name__ == "__main__":
    num_args = len(sys.argv)
    args = sys.argv
    if num_args != 3:
        print('TypeError: it takes exactly 2 arguments ({} given)'.format(num_args - 1))
        print(' '*5, 'Example usage: organise_ddsm.py input-dir output-dir')
        sys.exit(0)
    
    inp_dir = args[1]
    out_dir = args[2]

    print('Tracing through {}'.format(inp_dir))
    logs = list()  # to be passed by reference
    walk(inp_dir, out_dir, level=0, is_gt_folder=False, logs=logs)
    # print(logs)
    with open('logs.pkl', 'wb') as f:
        pickle.dump(logs, f)
    print('Fixing failed ones')
    fix_tmps(logs)

    print('Done')

        
