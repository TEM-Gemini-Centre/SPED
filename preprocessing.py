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
import matplotlib.pyplot as plt
import pickle
from tabulate import tabulate

from pathlib import Path

from scipy.ndimage import gaussian_filter
from skimage.exposure import rescale_intensity
from skimage.morphology import erosion
from skimage.morphology import disk
import skimage.filters as skifi


def load_pickle(filename):
    with open(filename, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def subtract_background_dog(array, sigma_min, sigma_max):
    """
    Subtract the background of a 2D array using difference of gaussians (DOG)

    :param array: The image
    :param sigma_min: The lower sigma
    :param sigma_max: The upper sigma
    :return:
    """
    blur_max = gaussian_filter(array, sigma_max)
    blur_min = gaussian_filter(array, sigma_min)
    return np.maximum(np.where(blur_min > blur_max, array, 0) - blur_max, 0)


def process_image(image, subtract_min=False, background_subtract_function=None, background_subtract_kwargs=None, sigma=None,
                  gamma=None, erode=False, footprint=disk(3), rescale=False, out_range=(0, 1), plot_steps=False,
                  **kwargs):
    """
    Process an image for templatematching

    Applies a series of image processing steps in the following order:
     - typecasting to float32
     - subtraction of minimum
     - background subtraction
     - gaussian blur
     - gamma correction
     - erosion
     - rescaling
     - typecasting to signed int16

    :param image: the image to process
    :param subtract_min: Whether to subtract the minimum intensity from the image
    :param background_subtract_function: Function to use for subtracting background
    :param background_subtract_kwargs: Keyword arguments passed to background subtract function
    :param sigma: sigma used in gaussian blurring
    :param erode: Whether to erode the image or not
    :param footprint: The erosion footprint, typically disk(3) or similar
    :param rescale: Whether to rescale the processed image
    :param out_range: The rescaling output range
    :param plot_steps: Whether to plot the steps or not.
    :param kwargs: Optional keyword arguments passed to matplotlib.pyplot.imshow if steps are plotted
    :type image: numpy.ndarray
    :type subtract_min: bool
    :type background_subtract_function: function
    :type background_subtract_kwargs: dict
    :type sigma: Union[int, float]
    :type gamma: Union[int, float]
    :type erode: bool
    :type footprint: numpy.ndarray
    :type rescale: bool
    :type out_range: 2-tuple
    :type plot_steps: bool
    """
    if plot_steps:
        images = {'raw': image}

    image = np.float32(image)
    if subtract_min:
        image = image - np.min(image)
        if plot_steps:
            images['min'] = image

    if background_subtract_function is not None:
        if background_subtract_kwargs is None:
            background_subtract_kwargs = {}
        image = background_subtract_function(image, **background_subtract_kwargs)
        if plot_steps:
            images['bcknd'] = image

    if sigma is not None:
        image = skifi.gaussian(image, sigma=sigma)
        if plot_steps:
            images[f'$\sigma=${sigma}'] = image

    if gamma is not None:
        image = image ** gamma
        if plot_steps:
            images[f'$\gamma$={gamma}'] = image

    if erode:
        image = erosion(image, footprint=footprint)
        if plot_steps:
            images['erosion'] = image

    if rescale:
        image = rescale_intensity(image, in_range='image', out_range=out_range)
        if plot_steps:
            images[f'rescaled {out_range}'] = image

    image = np.int16(image)
    if plot_steps:
        images[f'int16 ({np.min(image)}, {np.max(image)})'] = image

    if plot_steps:
        fig, axes = plt.subplots(nrows=1, ncols=len(images), figsize=(3 * len(images), 3), sharex=True, sharey=True,
                                 subplot_kw={'xticks': [], 'yticks': []})
        for ax, label in zip(axes, images):
            ax.imshow(images[label], **kwargs)
            ax.set_title(label)
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', type=Path, help='Path to a 4D-STEM dataset to templatematch')
    parser.add_argument('--map_min_to_zero', action='store_true', help='Whether to subtract the minimum from images')
    parser.add_argument('--background_subtract', action='store_true', help='Whether to subtract background using DOG')
    parser.add_argument('--min_sigma', type=float, default=3, help='Minimum sigma to use in background subtraction DOG')
    parser.add_argument('--max_sigma', type=float, default=8, help='Maximum sigma to use in background subtraction DOG')
    parser.add_argument('--blur_sigma', type=float, help='Sigma used in gaussian blurring')
    parser.add_argument('--gamma', type=float, help='Gamma value used to gamma-scale the images')
    parser.add_argument('--rescale', action='store_true',
                        help='Whether to rescale intensities after preprocessing')
    parser.add_argument('--rescale_range', type=float, nargs=2, default=[-0.02 * (2 ** 15 + 1), 1.0 * 2 ** 15 - 1], help='The output intensity range')
    parser.add_argument('--erode', action='store_true', help='Whether to erode the data using a footprint or not')
    parser.add_argument('--footprint', type=int, default=3, help='The erosion footprint')
    parser.add_argument('--lazy', action='store_true', help='Whether to work on a lazy signal or not')
    parser.add_argument('-v', '--verbose', dest='verbosity', default=0, action='count', help='Set verbose level')
    parser.add_argument('--no_overwrite', action='store_false',
                        help='Whether to not overwrite any existing preprocessed data')

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

    if arguments.background_subtract:
        background_subtract_function = subtract_background_dog
    else:
        background_subtract_function = None
    # Preprocessing
    preprocessing_kwargs = {
        'subtract_min': arguments.map_min_to_zero,
        'background_subtract_function': background_subtract_function,
        'background_subtract_kwargs': {
            'sigma_min': arguments.min_sigma,
            'sigma_max': arguments.max_sigma},
        'sigma': arguments.blur_sigma,
        'gamma': arguments.gamma,
        'erode': arguments.erode,
        'footprint': disk(arguments.footprint),
        'rescale': arguments.rescale,
        'out_range': arguments.rescale_range,
        'plot_steps': False
    }

    table = tabulate([[key, preprocessing_kwargs[key]] for key in preprocessing_kwargs], headers=['Parameter', 'Value'])
    logger.info(f'Preprocessing arguments:\n{table}')

    # Apply processing
    logger.debug(f'Copying signal')
    preprocessed_signal = signal.deepcopy()
    logger.info(f'Running preprocessing on {signal}')
    preprocessed_signal.map(process_image, **preprocessing_kwargs)
    if arguments.lazy:
        logger.info(f'Computing lazy signal')
        preprocessed_signal.compute()
    logger.info(f'Finished preprocessing')

    output_path = input_name.with_name(f'{input_name.stem}_preprocessed{input_name.suffix}')
    logger.info(f'Saving signal to {output_path.absolute()}')
    if arguments.lazy:
        preprocessed_signal.save(output_path, chunks=signal.data.chunksize, overwrite=arguments.no_overwrite)
    else:
        preprocessed_signal.save(output_path, overwrite=arguments.no_overwrite)

    logger.info('Finished preprocessing script')
