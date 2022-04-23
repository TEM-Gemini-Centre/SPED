import logging

# basic config must be done before loading other packages
# logger.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%d-%m-%Y %H:%M:%S')
logger = logging.getLogger(__file__)
logger.propagate = False
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%d-%m-%Y %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)

import argparse
from pathlib import Path
import hyperspy.api as hs
import numpy as np
import pickle
from tabulate import tabulate

from pathlib import Path

from scipy.ndimage import gaussian_filter
from skimage.exposure import rescale_intensity
import skimage.filters as skifi


def load_pickle(filename):
    with open(filename, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def subtract_background_dog(z, sigma_min, sigma_max):
    blur_max = gaussian_filter(z, sigma_max)
    blur_min = gaussian_filter(z, sigma_min)
    return np.maximum(np.where(blur_min > blur_max, z, 0) - blur_max, 0)


def process_image(image, map_min_to_zero=True, background_subtract_function=None, blur_sigma=None, min_intensity=None,
                  gamma_value=None, rescale=True, stripes=False, sigma_min=3, sigma_max=8):
    if stripes:
        # remove the vertical stripes artifact by subtracting the median of each column
        median_cols = np.median(image, axis=0)
        image = image - median_cols

    if map_min_to_zero:
        # map the minimum to 0
        image = image - image.min()

    if background_subtract_function is not None:
        # remove the entire diffuse background
        image = background_subtract_function(image, sigma_min, sigma_max)

    if blur_sigma is not None:
        # smooth the image
        image = skifi.gaussian(image, sigma=blur_sigma)

    if min_intensity is not None:
        # remove low intensity pixels
        image[image < min_intensity] = 0

    if gamma_value is not None:
        # change the gamma of the images
        image = image ** gamma_value

    if rescale:
        # remap intensities to 0-1 range
        image = rescale_intensity(image)
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', type=Path, help='Path to a 4D-STEM dataset to templatematch')
    parser.add_argument('--map_min_to_zero', action='store_true', help='Whether to subtract the minimum from images')
    parser.add_argument('--background_subtract', action='store_true', help='Whether to subtract background using DOG')
    parser.add_argument('--min_sigma', type=float, default=3, help='Minimum sigma to use in background subtraction DOG')
    parser.add_argument('--max_sigma', type=float, default=8, help='Maximum sigma to use in background subtraction DOG')
    parser.add_argument('--blur_sigma', type=float, help='Sigma used in gaussian blurring')
    parser.add_argument('--min_intensity', type=float,
                        help='Lower intensity threshold. Intensities below this value will be set to zero.')
    parser.add_argument('--gamma', type=float, help='Gamma value used to gamma-scale the images')
    parser.add_argument('--rescale', action='store_true',
                        help='Whether to remap intensities to [0, 1] after preprocessing')
    parser.add_argument('--stripes', action='store_true',
                        help='Whether to remove vertical stripes in the diffraction pattern.')
    parser.add_argument('--lazy', action='store_true', help='Whether to work on a lazy signal or not')
    parser.add_argument('-v', '--verbose', dest='verbosity', default=0, action='count', help='Set verbose level')

    arguments = parser.parse_args()

    log_level = [logging.WARNING, logging.INFO, logging.DEBUG][min([arguments.verbosity, 2])]
    logger.setLevel(log_level)

    args_as_str = [f'\n\t{arg!r} = {getattr(arguments, arg)}' for arg in vars(arguments)]
    logger.debug(f'Running template matching script with arguments:{"".join(args_as_str)}')

    # Load signal
    input_name = Path(arguments.filename)
    logger.info(f'Loading "{input_name.absolute()}" with lazy={arguments.lazy}')
    signal = hs.load(input_name, lazy=arguments.lazy)
    logger.debug(f'Loaded signal {signal}')

    # Preprocessing
    preprocessing_kwargs = {
        'map_min_to_zero': arguments.map_min_to_zero,
        'background_subtract_function': subtract_background_dog,
        'blur_sigma': arguments.blur_sigma,
        'min_intensity': arguments.min_intensity,
        'gamma_value': arguments.gamma,
        'rescale': arguments.rescale,
        'stripes': arguments.stripes,
        'sigma_min': arguments.min_sigma,
        'sigma_max': arguments.max_sigma
    }

    table = tabulate([[key, preprocessing_kwargs[key]] for key in preprocessing_kwargs], headers=['Parameter', 'Value'])
    logger.info(f'Preprocessing arguments:\n{table}')

    # Apply processing
    logger.debug(f'Copying signal')
    preprocessed_signal = signal.deepcopy()
    logger.debug(f'Changing datatype to float32')
    preprocessed_signal.change_dtype(np.float32)
    logger.info(f'Running preprocessing on {signal}')
    preprocessed_signal.map(preprocessed_signal, **preprocessing_kwargs)
    logger.info(f'Added preprocessing')

    output_path = input_name.with_name(f'{input_name.stem}_preprocessed{input_name.suffix}')
    logger.info(f'Saving signal to {output_path.absolute()}')
    if arguments.lazy:
        preprocessed_signal.save(output_path, chunks=signal.data.chunksize)
    else:
        preprocessed_signal.save(output_path)

    logger.info('Finished preprocessing script')
