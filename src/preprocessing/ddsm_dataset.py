import os
import sys
from matplotlib.pyplot import title
import numpy as np
from pathlib import Path
import yaml
import png
import math
import cv2
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from src.data_handling.ddsm_dataloader import *

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

def select_largest_obj(img_bin, lab_val=255, fill_holes=False, 
                        smooth_boundary=False, kernel_size=15, opening=False):
    '''Select the largest object from a binary image and optionally
    fill holes inside it and smooth its boundary.
    Args:
        img_bin (2D array): 2D numpy array of binary image.
        lab_val ([int]): integer value used for the label of the largest 
                object. Default is 255.
        fill_holes ([boolean]): whether fill the holes inside the largest 
                object or not. Default is false.
        smooth_boundary ([boolean]): whether smooth the boundary of the 
                largest object using morphological opening or not. Default 
                is false.
        kernel_size ([int]): the size of the kernel used for morphological 
                operation. Default is 15.
    Returns:
        a binary image as a mask for the largest object.
    '''
    if opening:
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, np.ones((10,150), np.uint8))
    # plt.figure()
    # plt.title('img_bin')
    # plt.imshow(img_bin, cmap=plt.cm.gray)
    # plt.show()
    n_labels, img_labeled, lab_stats, _ = \
        cv2.connectedComponentsWithStats(img_bin, connectivity=8, 
                                            ltype=cv2.CV_32S)
    largest_obj_lab = np.argmax(lab_stats[1:, 4]) + 1
    largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)
    largest_mask[img_labeled == largest_obj_lab] = lab_val
    # plt.figure()
    # plt.title('Before')
    # plt.imshow(largest_mask, cmap=plt.cm.gray)
    # plt.show()
    # import pdb; pdb.set_trace()
    if fill_holes:
        bkg_locs = np.where(img_labeled == 0)
        bkg_seed = (bkg_locs[0][0], bkg_locs[1][0])
        img_floodfill = largest_mask.copy()
        h_, w_ = largest_mask.shape
        mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)
        cv2.floodFill(img_floodfill, mask_, seedPoint=bkg_seed, 
                        newVal=lab_val)
        holes_mask = cv2.bitwise_not(img_floodfill)  # mask of the holes.
        largest_mask = largest_mask + holes_mask
    if smooth_boundary:
        # kernel_ = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        # largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN, 
        #                                 kernel_, iterations=5)
        largest_mask = cv2.dilate(largest_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(kernel_size,kernel_size)), iterations=8)
        
    # plt.figure()
    # plt.title('Largest Mask')
    # plt.imshow(largest_mask, cmap=plt.cm.gray)
    # plt.show()    
    return largest_mask

def max_pix_val(dtype):
    if dtype == np.dtype('uint8'):
        maxval = 2**8 - 1
    elif dtype == np.dtype('uint16'):
        maxval = 2**16 - 1
    else:
        raise Exception('Unknown dtype found in input image array')
    return maxval

def suppress_artifacts(img, global_threshold=.05, fill_holes=False, 
                        smooth_boundary=True, kernel_size=15, plot=False, opening=False):
    '''Mask artifacts from an input image
    Artifacts refer to textual markings and other small objects that are 
    not related to the breast region.
    Args:
        img (2D array): input image as a numpy 2D array.
        global_threshold ([int]): a global threshold as a cutoff for low 
                intensities for image binarization. Default is 18.
        kernel_size ([int]): kernel size for morphological operations. 
                Default is 15.
    Returns:
        a tuple of (output_image, breast_mask). Both are 2D numpy arrays.
    '''
    maxval = max_pix_val(img.dtype)
    if global_threshold < 1.:
        low_th = int(255*global_threshold)
    else:
        low_th = int(global_threshold)
    _, img_bin = cv2.threshold(img, low_th, maxval=255, type=cv2.THRESH_BINARY)
    breast_mask = select_largest_obj(img_bin, lab_val=maxval, 
                                            fill_holes=fill_holes, 
                                            smooth_boundary=smooth_boundary, 
                                            kernel_size=kernel_size,
                                            opening=opening)
    # edges = cv2.Canny(breast_mask, 150, 255)
    adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type = cv2.THRESH_BINARY_INV
    bin_img = cv2.adaptiveThreshold(breast_mask, 255, adapt_type, thresh_type, 11, 2)
    if plot:
        plt.figure()
        plt.title('adaptiveThreshold')
        plt.imshow(bin_img, cmap=plt.cm.gray)
    # Run Hough on edge detected image
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = min(200, max(img.shape)*0.15)  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on
    lines = cv2.HoughLinesP(bin_img, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    x_values = []
    y_values = []
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                if angle < 0:
                    angle += 180
                if img.shape[1] > img.shape[0]:
                    if (angle > 80 and angle < 100):
                        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0), 2)
                        if np.std([x1,x2]) < np.std([y1,y2]):
                            x_values.append(x1)
                            x_values.append(x2)
                        else:
                            y_values.append(y1)
                            y_values.append(y2)
                    # else:
                    #     print(angle)
                else:
                    if (angle < 10 or angle > 170):
                        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0), 2)
                        if np.std([x1,x2]) < np.std([y1,y2]):
                            x_values.append(x1)
                            x_values.append(x2)
                        else:
                            y_values.append(y1)
                            y_values.append(y2)                      
                # if (angle < 10 or angle > 170) or (angle > 80 and angle < 100):
                #     cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0), 2)
                #     if np.std([x1,x2]) < np.std([y1,y2]):
                #         x_values.append(x1)
                #         x_values.append(x2)
                #     else:
                #         y_values.append(y1)
                #         y_values.append(y2)
        margin = 0
        x = [a for a in x_values if a > int(img.shape[1]*0.85)]
        if not x:
            x_max = img.shape[1]
        else:
            x_max = min(x) - margin + 1
        x = [a for a in x_values if a < int(img.shape[1]*0.15)]
        if not x:
            x_min = 0
        else:
            x_min = max(x) + margin
        y = [a for a in y_values if a > int(img.shape[0]*0.85)]
        if not y:
            y_max = img.shape[0]
        else:
            y_max = min(y) - margin + 1
        y = [a for a in y_values if a < int(img.shape[0]*0.15)]
        if not y:
            y_min = 0
        else:
            y_min = max(y) + margin
        (x0, x1, y0, y1) = (y_min, y_max, x_min, x_max)
    else:
        (x0, x1, y0, y1) = (0, img.shape[0], 0, img.shape[1])
    if plot:
        plt.figure()
        plt.imshow(line_image)
        cv2.rectangle(line_image,(y0, x0), (y1, x1), (255,255,0), 3)
        plt.figure()
        plt.imshow(line_image)
        plt.show()
    img_suppr = cv2.bitwise_and(img, breast_mask)
    # OpenCV coordinates are YX
    cropped_image = img_suppr[x0:x1, y0:y1]
    # if plot:
    #     plt.figure()
    #     plt.imshow(img, cmap=plt.cm.gray)
    #     plt.figure()
    #     plt.imshow(cropped_image, cmap=plt.cm.gray)
    #     plt.show()

    return (cropped_image, breast_mask, (x0, x1, y0, y1))


def preprocessed_ddsm_image(img_path, mask_path, case_id, output_folder, save_masked=False, plot=False, opening=False):

    png_img_path = os.path.join(output_folder, case_id)
    png_mask_path = os.path.join(output_folder, case_id + '_mask')
    scaler = MinMaxScaler(feature_range=(0,255))

    nii_mask = nib.load(str(mask_path))
    ndarray_nii_mask = np.array(nii_mask.get_fdata()).copy()

    mask_norm = np.uint8(scaler.fit_transform(np.squeeze(ndarray_nii_mask)))
    nii = nib.load(str(img_path))
    ndarray_nii = np.array(nii.get_fdata()).copy()
    # scaled = StandardScaler().fit_transform(np.squeeze(ndarray_nii).copy())
    scaled = cv2.convertScaleAbs(np.squeeze(ndarray_nii), alpha=(255.0/65535.0))
    if plot:
        plt.figure()
        plt.imshow(mask_norm, cmap=plt.cm.gray)
        plt.figure()
        plt.imshow(scaled, cmap=plt.cm.gray)
        plt.show()
    img_suppr, breast_mask, cropped_coordinates = suppress_artifacts(scaled, global_threshold=.05, fill_holes=False, 
                        smooth_boundary=True, kernel_size=30, plot=plot, opening=opening)
    (x0, x1, y0, y1) = cropped_coordinates
    mask_norm = mask_norm[x0:x1, y0:y1]
    # Bitwise and Nifti Image and mask artifacts
    # if plot:
    #     plt.figure()
    #     plt.imshow(breast_mask, cmap=plt.cm.gray)
    #     plt.show()
    # Bitwise AND and save
    # input_nii = np.squeeze(ndarray_nii).copy()
    # cropped_breast_mask = breast_mask[x0:x1, y0:y1]
    # cropped_input_nii = input_nii[x0:x1, y0:y1]
    # img_nii_masked = (cropped_input_nii + cropped_breast_mask) - (cropped_input_nii*cropped_breast_mask)
    # # Update the affine matrix
    # affine = nii.affine
    # new_origin = np.dot(affine, np.array([x0, y0, 0, 1]))[:3]
    # affine[:3, 3] = new_origin
    # img_masked_niftii = os.path.join(output_folder, case_id + '_masked_crop_2.nii.gz')
    # new_img_nii_masked = nib.Nifti1Image(img_nii_masked, affine, nii.header)
    # nib.save(new_img_nii_masked, img_masked_niftii)
    # python nii2png.py -i /home/lidia/Datasets/TCIA/DDSM/CBIS-DDSM-NIFTI/Test/P_00066/LEFT_CC/scan.nii.gz -o /home/lidia/Datasets/TCIA/DDSM/CBIS-DDSM-NIFTI/processed/Nifti/masked/P_00066_LEFT_CC_mass_1.png
    img_breast, breast_mask, crop_roi = segment_breast(img_suppr, low_int_threshold=.05, crop=True)
    (x0, x1, y0, y1) = crop_roi
    mask_norm = mask_norm[x0:x1, y0:y1]
    if plot:
        plt.figure()
        plt.imshow(img_breast, cmap=plt.cm.gray)
        plt.figure()
        plt.imshow(mask_norm, cmap=plt.cm.gray)
        plt.show()

    if save_masked:
        contours, hierarchy = cv2.findContours(mask_norm.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cont_areas = [ cv2.contourArea(cont) for cont in contours ]
        idx = np.argmax(cont_areas) 
        cont = cv2.cvtColor(img_breast.copy(), cv2.COLOR_GRAY2RGB)
        for idx, countor in enumerate(cont_areas):
            cont = cv2.drawContours(cont, contours, idx, (255,0,0), 1)
        if plot:
            plt.figure()
            plt.imshow(cont)
            plt.show()
        # Make output folders
        masked_img_folder = os.path.join(str(output_folder),'masked_calc')
        if not Path(masked_img_folder).exists():
            Path(masked_img_folder).mkdir(parents=True)
        masked_img_path = os.path.join(masked_img_folder, case_id)
        # Write the Mask PNG file
        with open(masked_img_path + '.png', 'wb') as png_file:
            image_2d = cont.reshape(-1, cont.shape[1]*3)
            w = png.Writer(cont.shape[1], cont.shape[0], greyscale=False)
            w.write(png_file, image_2d)
    else:
        # Write the Scan PNG file
        with open(png_img_path + '.png', 'wb') as png_file:
            w = png.Writer(img_breast.shape[1], img_breast.shape[0], greyscale=True)
            w.write(png_file, img_breast.copy())

        # Write the Mask PNG file
        with open(png_mask_path + '.png', 'wb') as png_file:
            w = png.Writer(mask_norm.shape[1], mask_norm.shape[0], greyscale=True)
            w.write(png_file, mask_norm.copy())

    return png_img_path, png_mask_path

def save_patch_images(dataset_ddsm, out_dir):
    
    with tqdm(total=len(df_dataset)) as pbar:
        for case in df_dataset:
            if case['type'] == 'mass':
                img_roi_path = case['cropped_roi']
                if case['pathology'] == 'MALIGNANT':
                    out_path = os.path.join(out_dir, 'Malignant_mass')
                else:
                    out_path = os.path.join(out_dir, 'Benign_mass')
                if not Path(out_path).exists():
                    Path(out_path).mkdir(parents=True)
                out_path = os.path.join(out_path, case['case_id'])
                nii = nib.load(str(img_roi_path))
                ndarray_nii = np.array(nii.get_fdata()).copy()
                # scaled = StandardScaler().fit_transform(np.squeeze(ndarray_nii).copy())
                scaled = cv2.convertScaleAbs(np.squeeze(ndarray_nii), alpha=(255.0/65535.0))
                # Write the Scan PNG file
                with open(out_path + '.png', 'wb') as png_file:
                    w = png.Writer(scaled.shape[1], scaled.shape[0], greyscale=True)
                    w.write(png_file, scaled.copy())

if __name__ == '__main__':

    if len(sys.argv) > 1:
        config_file = Path(sys.argv[1])
    else:
        config_file = Path(
            '/home/lidia/source/BreastCancer/src/configs/preprocess_ddsm.yaml')
    with open(config_file) as file:
        config = yaml.safe_load(file)

    dataset = config['dataset']
    plot_show = config['plot']['show']
    save_masked = config['plot']['save_masked']
    dataset_path = config['paths']['dataset']
    info_csv = config['paths']['info_csv']
    output_folder = config['paths']['output_folder']

    # Make output folders
    if not Path(output_folder).exists():
        Path(output_folder).mkdir(parents=True)

    if dataset == 'DDSM':
        df_dataset = DDSMDataset(info_file=info_csv, dataset_path=dataset_path)
    else:
        print('Wrong dataset')
        exit(0)

    print(dataset + ' dataset cases: ' + str(len(df_dataset)))

    #selected_case = 'P_00649_LEFT_MLO_calcification_1'
    apply_opening = ['P_01167_LEFT_MLO_mass_1']
    selected_case = None
    with tqdm(total=len(df_dataset)) as pbar:
        for case in df_dataset:
            if (selected_case is None) or (case['case_id'] == selected_case):
                if case['case_id'] in apply_opening:
                    opening = True
                else:
                    opening = False
                img_scan_path = case['scan']
                msk_scan_path = case['mask']
                img_norm_path, msk_norm_path = preprocessed_ddsm_image(img_scan_path, msk_scan_path, case['case_id'], 
                                output_folder=output_folder, save_masked=save_masked, plot=plot_show, opening=opening)
                pbar.update(1)
    
    # Save mass patches  
    # out_dir = '/home/lidia/Datasets/TCIA/DDSM/CBIS-DDSM-NIFTI/patches/train'
    # info_csv = '/home/lidia/Datasets/TCIA/DDSM/report_processed_mass_case_description_train_set_train_split.csv'
    # df_dataset = DDSMDataset(info_file=info_csv, dataset_path=dataset_path)
    # save_patch_images(df_dataset, out_dir)
    # out_dir = '/home/lidia/Datasets/TCIA/DDSM/CBIS-DDSM-NIFTI/patches/val'
    # info_csv = '/home/lidia/Datasets/TCIA/DDSM/report_processed_mass_case_description_train_set_val_split.csv'
    # df_dataset = DDSMDataset(info_file=info_csv, dataset_path=dataset_path)
    # save_patch_images(df_dataset, out_dir)
    # out_dir = '/home/lidia/Datasets/TCIA/DDSM/CBIS-DDSM-NIFTI/patches/test'
    # info_csv = '/home/lidia/Datasets/TCIA/DDSM/report_processed_mass_case_description_test_set.csv'
    # df_dataset = DDSMDataset(info_file=info_csv, dataset_path=dataset_path)
    # save_patch_images(df_dataset, out_dir)