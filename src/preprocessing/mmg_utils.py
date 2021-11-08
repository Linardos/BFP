import os
import nibabel as nib
import numpy as np
import cv2
import png
# Need to install: $ sudo apt-get install -y dcm2niix

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

def mmg_dicom_2_png(dicom_filepath, out_folder_path, out_scan_filename, crop_breast=True):
    # dicom_filepath = 'path/to/dicom/image.dcm'
    # out_folder_path = 'path/to/save/nifti/images'
    # out_scan_filename = 'image_filename_uid'
    # crop_breast = True # Crop images to breast region (discard background)

    # Convert DICOM to NIFTI
    out_nii_image_path = os.path.join(out_folder_path, out_scan_filename + '.nii.gz')
    # Save NIFTI image to disk the first time (mandatory)
    if not os.path.exists(out_nii_image_path):
        dcim2nii_cmd = "dcm2niix -s y -b n -f " + out_scan_filename + " -o " + out_folder_path + " -z y " + dicom_filepath
        os.system(dcim2nii_cmd)

    out_png_image_path = os.path.join(out_folder_path, out_scan_filename + '.png')
    if not os.path.exists(out_png_image_path):
        # Convert NIFTI to PNG
        nii = nib.load(out_nii_image_path)
        ndarray_nii = np.array(nii.get_fdata()).copy()
        max_pix_value = ndarray_nii.max()
        scaled_scan = cv2.convertScaleAbs(np.squeeze(ndarray_nii), alpha=(255.0/float(max(max_pix_value,1))))
        png_image  = np.transpose(scaled_scan, (1, 0))
        if crop_breast:
            png_image = np.flipud(png_image)
            png_image, (xmin, xmax, ymin, ymax) = fit_to_breast(png_image)
            # crop_coordinates = xmin, xmax, ymin, ymax
        else:
            png_image = np.flipud(png_image)
        with open(out_png_image_path, 'wb') as png_file:
            w = png.Writer(png_image.shape[1], png_image.shape[0], greyscale=True)
            w.write(png_file, png_image.copy())
        # print(f'Saved {out_png_image_path}')

    return out_png_image_path
