import os
import sys
from matplotlib.pyplot import title
import numpy as np
from pathlib import Path
import yaml
import png
import math
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import pandas as pd
from skimage.draw import line, polygon
from src.visualizations.plot_image import plot_image_opencv_fit_window

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


if __name__ == '__main__':

    if len(sys.argv) > 1:
        config_file = Path(sys.argv[1])
    else:
        config_file = Path(
            '/home/lidia/source/BreastCancer/src/configs/preprocess_bcdr.yaml')
    with open(config_file) as file:
        config = yaml.safe_load(file)

    dataset = config['dataset']
    plot_show = config['plot']['show']
    save_masked_scans = config['plot']['save_masked_scans']
    save_masks = config['plot']['save_masks']
    crop_breast = config['crop_breast']
    dataset_path_list = config['path_' + dataset]
    info_csv_list = config['info_csv_' + dataset]
    outlines_csv_list = config['outlines_csv_' + dataset]
    output_folder = config['output_folder']
    # Make output folders
    if not Path(output_folder).exists():
        Path(output_folder).mkdir(parents=True)


    for dataset_path, info_csv, outlines_csv in zip(dataset_path_list, info_csv_list, outlines_csv_list):    
        
        df = pd.DataFrame(columns=('patient_id', 'study_id', 'series', 'scan_path', 'laterality', 'view', 'density',
                                    'age', 'classification', 'scan_width', 'scan_height', 'breast_x1', 'breast_x2', 'breast_y1', 'breast_y2',
                                    'breast_width', 'breast_height', 'lesion_id', 'segmentation_id', 'lesion_pathologies',
                                    'mask_path', 'lesion_x1', 'lesion_x2', 'lesion_y1', 'lesion_y2', 'lw_x_points', 'lw_y_points'))
        info = pd.read_csv(info_csv, delimiter=',')
        info = info.astype(object).replace(np.nan, '')
        if outlines_csv:
            outlines = pd.read_csv(outlines_csv, delimiter=',')
            outlines = outlines.astype(object).replace(np.nan, '')
        
        row_ctr = 0
        output_path = os.path.join(output_folder, dataset_path.split('/')[-1])
        unique_image_id_df = info.groupby(['image_filename'], as_index=False)
        for unique_image_filename, image_info_group in unique_image_id_df:
            for idx_info, image_info_row in enumerate(image_info_group.itertuples()):
                image_filename = unique_image_filename
                image_filename = image_filename.replace(' ', '')
                scan_info = {}
                scan_info.update({'patient_id': image_info_row.patient_id})
                scan_info.update({'study_id': image_info_row.study_id})
                scan_info.update({'series': image_info_row.series})
                scan_info.update({'scan_path': image_filename.replace('tif', 'png')})
                if isinstance(image_info_row.density, str):
                    breast_density = image_info_row.density.replace(' ', '')
                    if breast_density in ['1', '2', '3', '4']:
                        scan_info.update({'density': int(breast_density)})
                    else:
                        scan_info.update({'density': 0})
                else:
                    scan_info.update({'density': int(image_info_row.density)})
                scan_info.update({'laterality': IMAGE_TYPE_DICT[int(image_info_row.image_type_id)][0]})
                scan_info.update({'view': IMAGE_TYPE_DICT[int(image_info_row.image_type_id)][1]})
                scan_info.update({'age': image_info_row.age})
                scan_info.update({'classification': 'Normal'})
                scan = cv2.imread(os.path.join(dataset_path, image_filename), cv2.IMREAD_UNCHANGED) #8bit
                # import tifffile as tiff
                # a = tiff.imread(os.path.join(dataset_path, image_filename))
                scan_info.update({'scan_height': scan.shape[0]})
                scan_info.update({'scan_width': scan.shape[1]})
                cropped_png_path = os.path.join(output_path, image_filename.replace('tif', 'png'))
                if not Path(os.path.dirname(cropped_png_path)).exists():
                    Path(os.path.dirname(cropped_png_path)).mkdir(parents=True)
                png_image, (xmin, xmax, ymin, ymax) = fit_to_breast(scan)
                scan_info.update({f'breast_x1': xmin})
                scan_info.update({f'breast_x2': xmax})
                scan_info.update({f'breast_y1': ymin})
                scan_info.update({f'breast_y2': ymax})
                if not Path(cropped_png_path).exists():
                    with open(cropped_png_path, 'wb') as png_file:
                        w = png.Writer(png_image.shape[1], png_image.shape[0], greyscale=True)
                        w.write(png_file, png_image.copy())
                    print(f'Saved {cropped_png_path}')
                scan_color = np.array(Image.open(cropped_png_path).convert('RGB'))
                scan_info.update({'breast_height': scan_color.shape[0]})
                scan_info.update({'breast_width': scan_color.shape[1]})
                if plot_show:
                    plot_image_opencv_fit_window(scan_color, title='BCDR Scan', screen_resolution=(1920, 1080),
                                        wait_key=True)
                #image_group_outlines = outlines.groupby(['image_filename'], as_index=False)
                if outlines_csv:
                    outline_rows = outlines.loc[outlines['image_filename'] == unique_image_filename]
                else:
                    outline_rows = []
                if len(outline_rows) > 0:
                    for idx_outline, outline in enumerate(outline_rows.itertuples()):
                        # Read segmentation
                        lw_x_points = list(map(int, outline.lw_x_points.split(" ")[1:]))
                        lw_y_points = list(map(int, outline.lw_y_points.split(" ")[1:]))
                        outline_lesion = []
                        for i in range(len(lw_x_points)):
                            outline_lesion.append([lw_x_points[i], lw_y_points[i]])
                        # Close polygon
                        outline_lesion.append([lw_x_points[0], lw_y_points[0]])
                        if save_masks:
                            lesion = np.array(outline_lesion)
                            mask = np.zeros(scan.shape, dtype=np.uint8)
                            rr, cc = polygon(lesion[:,1], lesion[:,0], mask.shape)
                            mask[rr,cc] = 255
                            # mask = np.zeros(scan.shape, dtype=np.uint8)
                            # mask = cv2.fillConvexPoly(mask, np.array(outline_lesion, 'int32'), 255)
                            mask = mask[ymin:ymax, xmin:xmax]
                            lesion_id = outline.lesion_id
                            mask_filename =  image_filename[:-4] + '_mask_id_' + str(lesion_id) + '.png'
                            #cv2.imwrite(os.path.join(dataset_path, mask_filename), mask)
                            cropped_mask_path = os.path.join(output_path, mask_filename)
                            if plot_show:
                                plot_image_opencv_fit_window(mask, title='BCDR Scan', screen_resolution=(1920, 1080),
                                                wait_key=True)
                            if not Path(cropped_mask_path).exists():
                                with open(cropped_mask_path, 'wb') as png_file:
                                    w = png.Writer(mask.shape[1], mask.shape[0], greyscale=True)
                                    w.write(png_file, mask.copy())
                                print(f'Saved {cropped_mask_path}')
                            mask_grey = np.array(Image.open(cropped_mask_path))
                            contours, hierarchy = cv2.findContours(mask_grey.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            cont_areas = [ cv2.contourArea(cont) for cont in contours ]
                            max_idx = np.argmax(cont_areas)  # find the largest contour, i.e. breast.
                            cont = scan_color.copy()
                            for idx, countor in enumerate(cont_areas):
                                cont = cv2.drawContours(cont, contours, idx, (0,0,255), 4)
                            rect = cv2.boundingRect(contours[max_idx])
                            x,y,w,h = rect
                            cv2.rectangle(cont, (x, y), (x + w, y + h), (0,255,0), 4)
                            if plot_show:
                                plot_image_opencv_fit_window(cont, title='BCDR Scan', screen_resolution=(1920, 1080),
                                            wait_key=True)
                            # if save_masked_scans:
                            #     cont_png_path = mask_filename.replace('.png', '_show.png')
                            #     if not Path(cont_png_path).exists():
                            #         with open(cont_png_path, 'wb') as png_file:
                            #             #image_2d = np.reshape(cont, (-1, cont.shape[1] * 3))
                            #             w = png.Writer(cont.shape[1], cont.shape[0], greyscale=False)
                            #             w.write(png_file, cont.copy())
                            #         print(f'Saved {cont_png_path}')
                            pathologies = []
                            if int(outline.mammography_nodule):
                                pathologies.append('nodule')
                            if int(outline.mammography_calcification):
                                pathologies.append('calcification')
                            if int(outline.mammography_microcalcification):
                                pathologies.append('microcalcification')
                            if int(outline.mammography_axillary_adenopathy):
                                pathologies.append('axillary_adenopathy')
                            if int(outline.mammography_architectural_distortion):
                                pathologies.append('architectural_distortion')
                            if int(outline.mammography_stroma_distortion):
                                pathologies.append('stroma_distortion')
                            scan_info.update({'lesion_id': outline.lesion_id})
                            scan_info.update({'segmentation_id': outline.segmentation_id})
                            scan_info.update({'lesion_pathologies': pathologies})
                            scan_info.update({'mask_path': mask_filename})
                            scan_info.update({'lesion_x1': x})
                            scan_info.update({'lesion_x2': x + w})
                            scan_info.update({'lesion_y1': y})
                            scan_info.update({'lesion_y2': y + h})
                            scan_info.update({'lw_x_points': outline.lw_x_points})
                            scan_info.update({'lw_y_points': outline.lw_y_points})
                            scan_info.update({'classification': outline.classification.replace(' ', '')})
                            df = df.append(scan_info, ignore_index=True)
                else:
                    df = df.append(scan_info, ignore_index=True)
        df.to_csv(os.path.join(output_path, 'dataset_info.csv'), index=False)

# if __name__ == '__main__':

#     if len(sys.argv) > 1:
#         config_file = Path(sys.argv[1])
#     else:
#         config_file = Path(
#             '/home/lidia/source/BreastCancer/src/configs/preprocess_bcdr.yaml')
#     with open(config_file) as file:
#         config = yaml.safe_load(file)

#     dataset = config['dataset']
#     plot_show = config['plot']['show']
#     save_masked_scans = config['plot']['save_masked_scans']
#     save_masks = config['plot']['save_masks']
#     dataset_path = config['paths']['dataset']
#     info_csv = config['paths']['info_csv']
#     outlines_csv = config['paths']['outlines_csv']
#     output_folder = config['paths']['output_folder']
    
#     # Make output folders
#     if not Path(output_folder).exists():
#         Path(output_folder).mkdir(parents=True)

#     outlines_csv = pd.read_csv(outlines_csv, delimiter=',')
#     outlines = outlines_csv.astype(object).replace(np.nan, '')
#     outlines.head()

#     counter = 0
#     count_max = outlines.shape[0]
#     with tqdm(total=outlines.shape[0]) as pbar:
#         for index, case in outlines.iterrows():
#             image_filename = case['image_filename']
#             image_filename = image_filename.replace(' ', '')
#             scan = cv2.imread(os.path.join(dataset_path, image_filename), cv2.IMREAD_UNCHANGED)
#             if plot_show:
#                 plt.figure()
#                 plt.imshow(scan, cmap=plt.cm.gray)
#                 plt.show()
#             lw_x_points = list(map(int, case['lw_x_points'].split(" ")[1:]))
#             lw_y_points = list(map(int, case['lw_y_points'].split(" ")[1:]))
#             outline_lesion = []
#             if case['classification'].replace(' ', '') == 'Malign':
#                 color = (0,0,255)
#             else:
#                 color = (0,255,0)
#             thickness = 2
#             scan_color = cv2.cvtColor(scan.copy(), cv2.COLOR_GRAY2BGR)
#             for i in range(len(lw_x_points)):
#                 if i == len(lw_x_points)-1:
#                     scan_color = cv2.line(scan_color, (lw_x_points[i], lw_y_points[i]), (lw_x_points[0], lw_y_points[0]), color, thickness)
#                 else:
#                     scan_color = cv2.line(scan_color, (lw_x_points[i], lw_y_points[i]), (lw_x_points[i+1], lw_y_points[i+1]), color, thickness)
#                 outline_lesion.append([lw_x_points[i], lw_y_points[i]])
#             # Close polygon
#             outline_lesion.append([lw_x_points[0], lw_y_points[0]])
#             if plot_show:
#                 plt.figure()
#                 plt.imshow(cv2.cvtColor(scan_color.copy(), cv2.COLOR_BGR2RGB))
#                 plt.show()
#             if save_masked_scans:
#                 cv2.imwrite(os.path.join(output_folder, image_filename.replace('/', '_')), scan_color)
#             if save_masks:
#                 lesion = np.array(outline_lesion)
#                 mask = np.zeros(scan.shape, dtype=np.uint8)
#                 rr, cc = polygon(lesion[:,1], lesion[:,0], mask.shape)
#                 mask[rr,cc] = 255
#                 # mask = np.zeros(scan.shape, dtype=np.uint8)
#                 # mask = cv2.fillConvexPoly(mask, np.array(outline_lesion, 'int32'), 255)
#                 lesion_id = case['lesion_id']
#                 cv2.imwrite(os.path.join(dataset_path, image_filename[:-4] + '_mask_id_' + str(lesion_id) + '.tif'), mask)
#                 if plot_show:
#                     plt.figure()
#                     plt.imshow(mask, cmap=plt.cm.gray)
#                     plt.show()
#             pbar.update(1)
#             counter +=1
#             if counter > count_max:
#                 break