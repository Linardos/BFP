

from pathlib import Path
from typing import Dict, Callable, Tuple, Sequence, Union, Optional
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_CUTOFF = 0.01, 0.99
STANDARD_RANGE = 0, 100
TypeLandmarks = Union[str, Dict[str, Union[str, np.ndarray]]]

def _standardize_cutoff(cutoff: np.ndarray) -> np.ndarray:
    """Standardize the cutoff values given in the configuration.
    Computes percentile landmark normalization by default.
    """
    cutoff = np.asarray(cutoff)
    cutoff[0] = max(0, cutoff[0])
    cutoff[1] = min(1, cutoff[1])
    cutoff[0] = np.min([cutoff[0], 0.09])
    cutoff[1] = np.max([cutoff[1], 0.91])
    return cutoff

def _get_average_mapping(percentiles_database: np.ndarray) -> np.ndarray:
    """Map the landmarks of the database to the chosen range.
    Args:
        percentiles_database: Percentiles database over which to perform the
            averaging.
    """
    # Assuming percentiles_database.shape == (num_data_points, num_percentiles)
    pc1 = percentiles_database[:, 0]
    pc2 = percentiles_database[:, -1]
    s1, s2 = STANDARD_RANGE
    slopes = (s2 - s1) / (pc2 - pc1)
    slopes = np.nan_to_num(slopes)
    intercepts = np.mean(s1 - slopes * pc1)
    num_images = len(percentiles_database)
    final_map = slopes.dot(percentiles_database) / num_images + intercepts
    return final_map


def _get_percentiles(percentiles_cutoff: Tuple[float, float]) -> np.ndarray:
    quartiles = np.arange(25, 100, 25).tolist()
    deciles = np.arange(10, 100, 10).tolist()
    all_percentiles = list(percentiles_cutoff) + quartiles + deciles
    percentiles = sorted(set(all_percentiles))
    return np.array(percentiles)


def get_hist_stand_landmarks(images_paths, cutoff: Optional[Tuple[float, float]] = None,
    output_path: Optional[str] = None) -> np.ndarray:

    quantiles_cutoff = DEFAULT_CUTOFF if cutoff is None else cutoff
    percentiles_cutoff = 100 * np.array(quantiles_cutoff)
    percentiles_database = []
    percentiles = _get_percentiles(percentiles_cutoff)
    for i, image_file_path in enumerate(tqdm(images_paths)):
        tensor = Image.open(image_file_path)
        #w, h = orig_image.size
        #tensor, _ = read_image(image_file_path)
        # mask = np.ones_like(tensor, dtype=bool)
        # mask = mask.numpy() > 0
        # array = tensor.numpy()
        # percentile_values = np.percentile(array[mask], percentiles)
        tensor = np.array(tensor)
        mask = np.ones_like(tensor, bool)
        mask[tensor == 0] = False
        #mask = mask.reshape(-1)
        percentile_values = np.percentile(tensor[mask], percentiles)
        percentiles_database.append(percentile_values)
    percentiles_database = np.vstack(percentiles_database)
    mapping = _get_average_mapping(percentiles_database)

    if output_path is not None:
        output_path = Path(output_path).expanduser()
        extension = output_path.suffix
        if extension == '.txt':
            modality = 'image'
            text = f'{modality} {" ".join(map(str, mapping))}'
            output_path.write_text(text)
        elif extension == '.npy':
            np.save(output_path, mapping)
    return mapping

def apply_hist_stand_landmarks(image, landmarks,
        cutoff: Optional[Tuple[float, float]] = None, epsilon: float = 1e-5):
    cutoff_ = DEFAULT_CUTOFF if cutoff is None else cutoff
    mapping = landmarks

    data = np.array(image)
    shape = data.shape
    data = data.reshape(-1).astype(np.float32)

    mask = np.ones_like(data, bool)
    # mask[data == 0] = False

    range_to_use = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12]
    quantiles_cutoff = _standardize_cutoff(cutoff_)
    percentiles_cutoff = 100 * np.array(quantiles_cutoff)
    percentiles = _get_percentiles(percentiles_cutoff)
    percentile_values = np.percentile(data[mask], percentiles)

    # Apply linear histogram standardization
    range_mapping = mapping[range_to_use]
    range_perc = percentile_values[range_to_use]
    diff_mapping = np.diff(range_mapping)
    diff_perc = np.diff(range_perc)

    # Handling the case where two landmarks are the same
    # for a given input image. This usually happens when
    # image background is not removed from the image.
    diff_perc[diff_perc < epsilon] = np.inf

    affine_map = np.zeros([2, len(range_to_use) - 1])

    # Compute slopes of the linear models
    affine_map[0] = diff_mapping / diff_perc

    # Compute intercepts of the linear models
    affine_map[1] = range_mapping[:-1] - affine_map[0] * range_perc[:-1]

    bin_id = np.digitize(data, range_perc[1:-1], right=False)
    lin_img = affine_map[0, bin_id]
    aff_img = affine_map[1, bin_id]
    new_img = lin_img * data + aff_img
    new_img = new_img.reshape(shape)
    new_img = new_img.astype(np.float32)
    #new_img = torch.as_tensor(new_img)
    return new_img

def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    from skimage import img_as_float, exposure
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    input_img = image[image > 0.01]
    ax_hist.hist(input_img.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf