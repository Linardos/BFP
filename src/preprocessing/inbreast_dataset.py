import os
import sys
from matplotlib.pyplot import title
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import png
import cv2
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from src.data_handling.inbreast_dataset import INbreastDataset
from src.visualizations.plot_image import plot_image_opencv_fit_window

def segment_breast(img, low_int_threshold=.05, crop=True):
    '''Perform breast segmentation
    Args:
        low_int_threshold([float or int]): Low intensity threshold to 
                filter out background. It can be a fraction of the max 
                intensity value or an integer intensity value.
        crop ([bool]): Whether or not to crop the image.
    Returns:
        An image of the segmented breast.
    NOTES: the low_int_threshold is applied to an image of dtype 'uint8',
        which has a max value of 255.
    '''
    img_8u = (img.astype('float32')/img.max()*255).astype('uint8')
    max_value = 255
    # Create img for thresholding and contours.
    if low_int_threshold < 1.:
        low_th = int(img_8u.max()*low_int_threshold)
    else:
        low_th = int(low_int_threshold)
    _, img_bin = cv2.threshold(
        img_8u, low_th, maxval=max_value, type=cv2.THRESH_BINARY)

    contours,_ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_areas = [ cv2.contourArea(cont) for cont in contours ]
    idx = np.argmax(cont_areas)  # find the largest contour, i.e. breast.
    # fill the contour.
    breast_mask = cv2.drawContours(np.zeros_like(img_bin), contours, idx, max_value, -1) 
    # segment the breast.
    img_breast_only = cv2.bitwise_and(img, img, mask=breast_mask)
    # plt.figure()
    # plt.imshow(breast_mask, cmap=plt.cm.gray)
    # plt.figure()
    # plt.imshow(img_breast_only, cmap=plt.cm.gray)
    # plt.show()
    x,y,w,h = cv2.boundingRect(contours[idx])
    if crop:
        img_breast_only = img_breast_only[y:y+h, x:x+w]
    (x0, x1, y0, y1) = (y, y + h, x, x + w)
    return img_breast_only, breast_mask, (x0, x1, y0, y1)

def fit_to_breast(img_pil):
    low_int_threshold = 0.05
    max_value = 255
    img_8u = (img_pil.astype('float32')/img_pil.max()*max_value).astype('uint8')
    low_th = int(max_value*low_int_threshold)
    _, img_bin = cv2.threshold(img_8u, low_th, maxval=max_value, type=cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_areas = [ cv2.contourArea(cont) for cont in contours ]
    idx = np.argmax(cont_areas)
    x,y,w,h = cv2.boundingRect(contours[idx])
    img_breast = img_pil[y:y+h, x:x+w]
    (xmin, xmax, ymin, ymax) = (x, x + w, y, y + h)
    return img_breast, (xmin, xmax, ymin, ymax)

def preprocessed_inbreast_image(dataset_folder, img_path, masks_list, output_folder, crop_breast=True, save_masked=False, plot=False):

    # if not Path(output_folder).exists():
    #     if save_masked:
    #         Path(output_folder).mkdir(parents=True)
    #     else:
    #         Path(os.path.join(output_folder, 'scans')).mkdir(parents=True)
    #         Path(os.path.join(output_folder, 'masses')).mkdir(parents=True)
    #         Path(os.path.join(output_folder, 'calc')).mkdir(parents=True)

    # scaler = MinMaxScaler(feature_range=(0,255))
    # if calc_mask_path:   
    #     nii_mask_calc = nib.load(str(calc_mask_path))
    #     ndarray_nii_mask_calc = np.array(nii_mask_calc.get_fdata()).copy()
    #     mask_calc = np.uint8(scaler.fit_transform(np.squeeze(ndarray_nii_mask_calc)))
    #     if plot:
    #         plt.figure()
    #         plt.imshow(mask_calc, cmap=plt.cm.gray)
    # if masses_mask_path:
    #     nii_mask = nib.load(str(masses_mask_path))
    #     ndarray_nii_mask = np.array(nii_mask.get_fdata()).copy()
    #     mask_masses = np.uint8(scaler.fit_transform(np.squeeze(ndarray_nii_mask)))
    #     if plot:
    #         plt.figure()
    #         plt.imshow(mask_masses, cmap=plt.cm.gray)
    scan_info = {}
    scan_png_path = img_path.replace(dataset_folder, output_folder).replace('nii.gz', 'png')
    if not Path(os.path.dirname(scan_png_path)).exists():
        Path(os.path.dirname(scan_png_path)).mkdir(parents=True)

    nii = nib.load(img_path)
    ndarray_nii = np.array(nii.get_fdata()).copy()
    inbreast_14_bit = 16384.0
    max_pix_value = ndarray_nii.max()
    scaled_scan = cv2.convertScaleAbs(np.squeeze(ndarray_nii), alpha=(255.0/float(max(max_pix_value,1))))
    # if plot:
    #     plt.figure()
    #     plt.imshow(scaled_scan, cmap=plt.cm.gray)
    #     plt.show()
    png_image  = np.transpose(scaled_scan, (1, 0))
    scan_info.update({'scan_height': png_image.shape[0]})
    scan_info.update({'scan_width': png_image.shape[1]})
    if crop_breast:
        png_image = np.flipud(png_image)
        png_image, (xmin, xmax, ymin, ymax) = fit_to_breast(png_image)
        scan_info.update({f'breast_x1': xmin})
        scan_info.update({f'breast_x2': xmax})
        scan_info.update({f'breast_y1': ymin})
        scan_info.update({f'breast_y2': ymax})
    else:
        png_image = np.flipud(png_image)
    
    if not Path(scan_png_path).exists():
        with open(scan_png_path, 'wb') as png_file:
            w = png.Writer(png_image.shape[1], png_image.shape[0], greyscale=True)
            w.write(png_file, png_image.copy())
        print(f'Saved {scan_png_path}')
    img = np.array(Image.open(scan_png_path).convert('RGB'))
    scan_info.update({'breast_height': img.shape[0]})
    scan_info.update({'breast_width': img.shape[1]})

    #cv2.imwrite('./test_1.png', img)
    if plot:
        plot_image_opencv_fit_window(img, title='INBreast Scan', screen_resolution=(1920, 1080),
                            wait_key=True)

    scaler = MinMaxScaler(feature_range=(0,255))
    for mask in masks_list:
        if mask['type'] == 'mass':
            mask_id = str(mask['mask_id'])
            mass_name = f'mass_{mask_id}'
            nii_mask = nib.load(mask['mask'])
            ndarray_nii_mask = np.array(nii_mask.get_fdata()).copy()
            mask_uint8 = np.uint8(scaler.fit_transform(np.squeeze(ndarray_nii_mask)))
            png_mask = np.flipud(np.transpose(mask_uint8, (1, 0)))
            if crop_breast:
                png_mask = png_mask[ymin:ymax, xmin:xmax]
            
            mask_png_path = mask['mask'].replace(dataset_folder, output_folder).replace('nii.gz', 'png')
            if not Path(mask_png_path).exists():
                with open(mask_png_path, 'wb') as png_file:
                    w = png.Writer(png_mask.shape[1], png_mask.shape[0], greyscale=True)
                    w.write(png_file, png_mask.copy())
                print(f'Saved {mask_png_path}')
            mask_grey = np.array(Image.open(mask_png_path))
            # if plot:
            #     plot_image_opencv_fit_window(mask_grey, title='INBreast Scan', screen_resolution=(1920, 1080),
            #                     wait_key=True)

            
            contours, hierarchy = cv2.findContours(mask_grey.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cont_areas = [ cv2.contourArea(cont) for cont in contours ]
            max_idx = np.argmax(cont_areas)  # find the largest contour, i.e. breast.
            cont = img.copy()
            for idx, countor in enumerate(cont_areas):
                cont = cv2.drawContours(cont, contours, idx, (0,0,255), 4)
            # if plot:
            #     plot_image_opencv_fit_window(cont, title='INBreast Scan', screen_resolution=(1920, 1080),
            #                 wait_key=True)
            rect = cv2.boundingRect(contours[max_idx])
            x,y,w,h = rect
            cv2.rectangle(cont, (x, y), (x + w, y + h), (0,255,0), 4)
            scan_info.update({f'{mass_name}_x1': x})
            scan_info.update({f'{mass_name}_x2': x + w})
            scan_info.update({f'{mass_name}_y1': y})
            scan_info.update({f'{mass_name}_y2': y + h})
            if save_masked:
                cont_png_path = mask_png_path.replace('.png', '_show.png')
                if not Path(cont_png_path).exists():
                    with open(cont_png_path, 'wb') as png_file:
                        #image_2d = np.reshape(cont, (-1, cont.shape[1] * 3))
                        w = png.Writer(cont.shape[1], cont.shape[0], greyscale=False)
                        w.write(png_file, cont.copy())
                    print(f'Saved {cont_png_path}')
            if plot:
                plot_image_opencv_fit_window(cont, title='INBreast Scan', screen_resolution=(1920, 1080),
                            wait_key=True)
    
    return scan_info
    # img_breast, breast_mask, crop_roi = segment_breast(scaled_scan, low_int_threshold=.05, crop=True)
    # (x0, x1, y0, y1) = crop_roi
    # if calc_mask_path:
    #     mask_calc = mask_calc[x0:x1, y0:y1]
    #     if plot:
    #         plt.figure()
    #         plt.imshow(mask_calc, cmap=plt.cm.gray)
    # if masses_mask_path:
    #     mask_masses = mask_masses[x0:x1, y0:y1]
    #     if plot:
    #         plt.figure()
    #         plt.imshow(mask_masses, cmap=plt.cm.gray)
    # if plot:
    #     plt.figure()
    #     plt.imshow(img_breast, cmap=plt.cm.gray)
    #     plt.show()

    # png_calc_path = None
    # png_masses_path = None
    # if save_masked and (calc_mask_path or masses_mask_path):
    #     if calc_mask_path:
    #         mask = mask_calc
    #         if masses_mask_path:
    #             mask = mask_calc | mask_masses
    #     else:
    #         mask = mask_masses
    #     contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     cont_areas = [ cv2.contourArea(cont) for cont in contours ]
    #     idx = np.argmax(cont_areas)  # find the largest contour, i.e. breast.
    #     cont = cv2.cvtColor(img_breast.copy(),cv2.COLOR_GRAY2RGB)
    #     for idx, countor in enumerate(cont_areas):
    #         cont = cv2.drawContours(cont, contours, idx, (255,0,0), 1)
    #     if plot:
    #         plt.figure()
    #         plt.imshow(cont)
    #         plt.show()
    #     # Write the Mask PNG file
    #     png_img_path = os.path.join(output_folder, case_id)
    #     with open(png_img_path + '.png', 'wb') as png_file:
    #         image_2d = np.reshape(cont, (-1, cont.shape[1] * 3))
    #         w = png.Writer(cont.shape[1], cont.shape[0], greyscale=False)
    #         w.write(png_file, image_2d)
    # elif save_masked:
    #     # Write the Mask PNG file
    #     png_img_path = os.path.join(output_folder, case_id)
    #     with open(png_img_path + '.png', 'wb') as png_file:
    #         w = png.Writer(img_breast.shape[1], img_breast.shape[0], greyscale=True)
    #         w.write(png_file, img_breast.copy())
    # elif not save_masked:
    #     png_img_path = os.path.join(os.path.join(output_folder, 'scans'), case_id)
    #     # Write the Scan PNG file
    #     with open(png_img_path + '.png', 'wb') as png_file:
    #         w = png.Writer(img_breast.shape[1], img_breast.shape[0], greyscale=True)
    #         w.write(png_file, img_breast.copy())
    #     if calc_mask_path:
    #         png_calc_path = os.path.join(os.path.join(output_folder, 'calc'), case_id)
    #         # Write the Mask PNG file
    #         with open(png_calc_path + '.png', 'wb') as png_file:
    #             w = png.Writer(mask_calc.shape[1], mask_calc.shape[0], greyscale=True)
    #             w.write(png_file, mask_calc.copy())
    #     if masses_mask_path:
    #         png_masses_path = os.path.join(os.path.join(output_folder, 'masses'), case_id)
    #         # Write the Mask PNG file
    #         with open(png_masses_path + '.png', 'wb') as png_file:
    #             w = png.Writer(mask_masses.shape[1], mask_masses.shape[0], greyscale=True)
    #             w.write(png_file, mask_masses.copy())
    #return png_img_path, png_masses_path, png_calc_path

def get_mask_roi(mask):
    img_8u = (mask.astype('float32')/mask.max()*255).astype('uint8')
    _, img_bin = cv2.threshold(
        img_8u, 0, maxval=255, type=cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_areas = [ cv2.contourArea(cont) for cont in contours ]
    idx = np.argmax(cont_areas)  # find the largest contour, i.e. breast.
    # plt.figure()
    # plt.imshow(img_breast_only, cmap=plt.cm.gray)
    # plt.show()
    x,y,w,h = cv2.boundingRect(contours[idx])
    (x0, x1, y0, y1) = (y, y + h, x, x + w)
    return (x0, x1, y0, y1)

def crop_patch_inbreast(img_path, mass_mask_path, out_patch, plot=False):

    if not Path(os.path.dirname(out_patch)).exists():
        Path(os.path.dirname(out_patch)).mkdir(parents=True)

    scaler = MinMaxScaler(feature_range=(0,255))
    nii_mask = nib.load(str(mass_mask_path))
    ndarray_nii_mask = np.array(nii_mask.get_fdata()).copy()
    mask_mass = np.uint8(scaler.fit_transform(np.squeeze(ndarray_nii_mask)))
    if plot:
        plt.figure()
        plt.imshow(mask_mass, cmap=plt.cm.gray)

    nii = nib.load(str(img_path))
    ndarray_nii = np.array(nii.get_fdata()).copy()
    inbreast_14_bit = 16384.0
    max_pix_value = ndarray_nii.max()
    scaled_scan = cv2.convertScaleAbs(np.squeeze(ndarray_nii), alpha=(255.0/float(max_pix_value)))
    if plot:
        plt.figure()
        plt.imshow(scaled_scan, cmap=plt.cm.gray)
        plt.show()

    (x0, x1, y0, y1) = get_mask_roi(mask_mass)
    roi_mass = scaled_scan[x0:x1, y0:y1]

    if plot:
        plt.figure()
        plt.imshow(roi_mass, cmap=plt.cm.gray)
        plt.show()
    # Write the Mask ROI PNG file
    with open(out_patch + '.png', 'wb') as png_file:
        w = png.Writer(roi_mass.shape[1], roi_mass.shape[0], greyscale=True)
        w.write(png_file, roi_mass.copy())
        print(f'Saved {out_patch}')

if __name__ == '__main__':

    if len(sys.argv) > 1:
        config_file = Path(sys.argv[1])
    else:
        config_file = Path(
            '/home/lidia/source/BreastCancer/src/configs/preprocess_inbreast.yaml')
    with open(config_file) as file:
        config = yaml.safe_load(file)

    dataset = config['dataset']
    plot_show = config['plot']['show']
    save_masked = config['plot']['save_masked']
    dataset_path = config['paths']['dataset']
    info_csv = config['paths']['info_csv']
    output_folder = config['paths']['output_folder']

    # Make output folder
    if not Path(output_folder).exists():
        Path(output_folder).mkdir(parents=True)

    if dataset == 'INbreast':
        df_dataset = INbreastDataset(info_file=info_csv, dataset_path=dataset_path)
    else:
        print('Wrong dataset')
        exit(0)
    
    # output_folder = '/home/lidia/Datasets/InBreast/Patches/mass'
    # selected_case = None
    # with tqdm(total=len(df_dataset)) as pbar:
    #     for case in df_dataset:
    #         img_scan_path = case['scan']
    #         masks = df_dataset.get_masks(img_scan_path)
    #         if masks:
    #             for mask in masks:
    #                 if mask['type'] == 'mass':
    #                     mask_mass_path = mask['mask']
    #                     birad = case['birad']
    #                     patch_name = case['case_id'] + '_mass_' + str(mask['mask_id'])
    #                     out_patch = os.path.join(output_folder, os.path.join(birad, patch_name))
    #                     crop_patch_inbreast(img_scan_path, mask_mass_path, out_patch)
                        
    #         # png_scan, png_masses, png_calc = preprocessed_inbreast_image(img_scan_path, masses_mask_path,
    #         #             case['case_id'], output_folder=output, save_masked=save_masked, plot=plot_show)
    #         pbar.update(1)


    info = pd.read_csv(info_csv)
    head = info.head()
    print(dataset + ' dataset cases: ' + str(len(df_dataset)))
    #selected_case = 'f571fd4e63c718e3_L_CC_scan_0'
    selected_case = None
    with tqdm(total=len(df_dataset)) as pbar:
        for case in df_dataset:
            if (selected_case is None) or (case['case_id'] == selected_case):
                row_idx = int(info.loc[info['File Name'] == int(case['filename'])].index.values)
                img_scan_path = case['scan']
                masks_list = df_dataset.get_masks(img_scan_path)
                # if case['patient_id'] == '024ee3569b2605dc':
                #     print(row_idx)
                #     print(case['patient_id'])
                #     print(case['scan'])
                #     print(len(masks_list))
                scan_info = preprocessed_inbreast_image(dataset_path, img_scan_path, masks_list, output_folder=output_folder, 
                                crop_breast=config['crop_breast'], save_masked=save_masked, plot=plot_show)
                for key in scan_info:
                    info.at[row_idx, key] = scan_info[key]
                pbar.update(1)
    info.to_csv(config['paths']['output_csv'], index=False)