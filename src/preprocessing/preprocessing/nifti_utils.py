import os
import warnings
import numpy as np
import nibabel as nib
import torchio as tio
import matplotlib.pyplot as plt
from scipy import stats


def normalize_nifti_per_label(img_path, mask_path, output_tag=''):
    subject = tio.Subject(
                img=tio.ScalarImage(img_path),
                mask=tio.LabelMap(mask_path),
                )
    transform = tio.ZNormalization(masking_method='mask')
    transformed = transform(subject)
    normalized_img_path = os.path.splitext(os.path.splitext(img_path)[0])[0] + output_tag + '.nii'
    transformed['img'].save(normalized_img_path, squeeze=False)
    normalized_mask_path = os.path.splitext(os.path.splitext(mask_path)[0])[0] + output_tag + '.nii'
    transformed['mask'].save(normalized_mask_path, squeeze=False)
    
    return normalized_img_path, normalized_mask_path


def plot_histogram(axis, tensor, num_positions=100, label=None, alpha=0.05, color=None):
    values = tensor.numpy().ravel()
    max_value = int(max(values))
    print('max value: %d' % (max_value))
    kernel = stats.gaussian_kde(values)
    positions = np.linspace(values.min(), values.max(), num=num_positions)
    histogram = kernel(positions)
    kwargs = dict(
        linewidth=1, color='black' if color is None else color, alpha=alpha)
    if label is not None:
        kwargs['label'] = label
    axis.plot(positions, histogram, **kwargs)
    return max_value


def plot_histogram_image_list(image_paths, save_path=None):
    paths = image_paths
    fig, ax = plt.subplots(dpi=100)
    max_values = []
    for path in paths:
        tensor = tio.ScalarImage(path).data
        max_value = plot_histogram(ax, tensor, color='red')
        max_values.append(max_value)
    ax.set_xlim(-100, max(max_values))
    ax.set_ylim(0, 0.004)
    ax.set_title('Original histograms of all samples')
    ax.set_xlabel('Intensity')
    ax.grid()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()


def nifti_4d_to_phase(filename, phase, phase_name, out_path, gt_img=False):
    img = nib.load(filename)
    imgs = nib.four_to_three(img)

    froot, ext = splitext(filename)
    if ext in ('.gz', '.bz2'):
        froot, ext = splitext(froot)
    if out_path is not None:
        pth, fname = psplit(froot)
        froot = pjoin(out_path, fname)

    desired_3d_img = imgs[phase]
    raw_img = desired_3d_img.get_fdata()

    desired_3d_img = nib.Nifti1Image(
        raw_img, desired_3d_img.affine, desired_3d_img.header)

    fname3d = '%s_%s.nii' % (froot, phase_name)
    nib.save(desired_3d_img, fname3d)
    return(fname3d)


def mask_2d_image_nifti(image_path, mask_path, masked_path=None):
    img = nib.load(image_path)
    img_phase = img.get_fdata()
    header_info = img.header
    mask_img = nib.load(mask_path).get_fdata()
    img_phase[mask_img < 1] = 0
    img_masked_niftii = nib.Nifti1Image(img_phase, img.affine, header_info)
    if masked_path is None:
        return img_masked_niftii
    else:
        nib.save(img_masked_niftii, masked_path)
        return


def show_nifti(image_path_or_image, colormap='gray'):
    try:
        from niwidgets import NiftiWidget
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            widget = NiftiWidget(image_path_or_image)
            widget.nifti_plotter(colormap=colormap)
    except Exception:
        if isinstance(image_path_or_image, nib.AnalyzeImage):
            nii = image_path_or_image
        else:
            image_path = image_path_or_image
            nii = nib.load(str(image_path))
        k = int(nii.shape[-1] / 2)
        plt.imshow(np.squeeze(nii.dataobj), cmap=colormap)
