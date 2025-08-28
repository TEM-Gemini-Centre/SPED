"""
Script to preprocess a SPED dataset. It normalizes and centers the data stack. It does not perform image preprocessing (see the "preprocessing.py" script for this instead).
"""
import logging

logger = logging.getLogger(__file__)
logger.propagate = False
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%d-%m-%Y %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)

fh = logging.FileHandler('preprocessing_log.txt', 'w')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

import argparse
import hyperspy.api as hs
import pyxem as pxm
from skimage.io import imsave
import numpy as np
from skimage.feature import blob_log
from pathlib import Path
from diffsims.utils.sim_utils import get_electron_wavelength


def preprocess(filename, lazy=True, com_mask=(128, 128, 12), width=20, calibration=None, formats=None):
    """
    Preprocess a 4D STEM data file

    :param filename: The path to the data file
    :param lazy: Whether to work lazily
    :param com_mask: The region of the diffraction pattern to calculate COM within given in pixel coordinates (x, y, r)
    :param width: The width of the navigation frame to use when estimating the linear descan shift.
    :param formats: Output file formats. If None, both .hspy and .zarr will be created
    :type filename: Union[str, Path]
    :type lazy: bool
    :type com_mask: Union[None, tuple]
    :type width: int
    :type formats: Union[None, tuple]
    :return:
    """
    logger.debug(
        f'Preprocessing function got the following arguments:\n\tfilename: {filename!r}\n\tlazy: {lazy!r}\n\tcom_mask: {com_mask!r}\n\tformats: {formats!r}')
    formats = formats if formats else ('.hspy', '.zspy')
    logger.debug(f'I will save preprocessed signals in these formats: {formats}')

    filename = Path(filename)
    logger.info(f'Loading data from {filename}')

    # Load data
    signal = hs.load(filename, lazy=lazy)
    logger.debug(f'Loaded data')
    if not isinstance(signal, pxm.signals.ElectronDiffraction2D):
        logger.warning(
            f'Only ElectronDiffraction2D signals can be proeprocessed. I got {signal!r} of type {type(signal)}')

    # Slice
    # Restrict data to ROI
    # logger.info(f'Slicing the dataset')
    # signal = signal.inav[0:512, 0:512]
    # logger.debug(f'Sliced the dataset to navigation shape {signal.axes_manager.navigation_shape}')

    # Centering
    # Center the dataset by estimating a linear shift from the centre of mass of the direct beam
    logger.info('Centering dataset')
    logger.debug(f'Calculating COM within {com_mask}')
    # Calculate COM
    com = signal.center_of_mass(mask=com_mask)
    # Create beam shift object
    beam_shift = pxm.signals.BeamShift(com.T)
    # Create navigation mask for which pixels to use when calculating the beam shift
    mask = hs.signals.Signal2D(np.zeros(signal.axes_manager.navigation_shape, dtype=bool).T).T
    if width <= np.min(signal.axes_manager.navigation_shape):
        # Set a `width` frame to True. This will make the beam shift dependent on the "outer" regions of the scan while preserving a sufficient number for statistics
        mask.inav[width:-width, width:-width] = True
    else:
        # Set all pixels to False.
        mask.inav[:, :] = False
    logger.debug(f'Estimating linear plane')
    # Estimate the beam shifts
    beam_shift.make_linear_plane(mask=mask)
    # We want to shift the patterns to the center
    beam_shift = beam_shift - (signal.axes_manager.signal_shape[0] // 2)
    logger.info(
        f'Beam shifts are within {float(beam_shift.min(axis=[0, 1, 2]).data)} pixels and {float(beam_shift.max(axis=[0, 1, 2]).data)} pixels')
    # Shift the patterns
    signal.shift_diffraction(beam_shift.isig[0], beam_shift.isig[1], inplace=True)
    # Add metadata
    signal.metadata.add_dictionary({
        'Preprocessing': {
            'Centering': {
                'COM': com,
                'COM_mask': {
                    'x': com_mask[0],
                    'y': com_mask[1],
                    'r': com_mask[2]
                },
                'Shifts': beam_shift,
                'shift_estimate_mask': mask
            }
        }
    })

    # Calibration
    # Calibration value from measuring distance between (-4, 0, 0) and (4, 0, 0) Al reflections
    # calibration = 0.00943  # Å^-1
    if calibration is not None:
        calibration=float(calibration)
        logger.info(f'Setting calibration to {calibration} Å^-1')
        signal.set_diffraction_calibration(calibration)

    # Binning
    # Bin the signal dimension by a factor 2
    # binning = (1, 1, 2, 2)
    # logger.info(f'Binning signal with scales {binning}')
    # signal = signal.rebin(scale=binning)

    # Preparing masks
    # Prepare masks used for later data processing
    # logger.info(f'Preparing masks')
    # image = signal.mean(axis=[0, 1])
    # minimum_r = 8  #

    # Set up mask arrays
    # nx, ny = image.axes_manager.signal_shape
    # mask = np.zeros((nx, ny), dtype=bool)
    # direct_beam_mask = np.zeros((nx, ny), dtype=bool)
    # cutoff_mask = np.zeros((nx, ny), dtype=bool)

    # Cuton / Cutoff
    # cutoff_hkl = np.array([2, 2, 0])  # Make a mask with cutoff at a given g-vector
    # cuton_mrad = 4  # Make a mask that cutsoff everything up a certain mrad
    # a = 4.04  # Lattice parameter of aluminium in Å
    # cutoff_g = np.sqrt(np.sum(cutoff_hkl ** 2 / a ** 2))
    # cuton_k = cuton_mrad / 1000 / get_electron_wavelength(image.metadata.Acquisition_instrument.TEM.beam_energy / 1000)
    # logger.info(
    #     f'Minimum scattering vector: {cuton_k} {image.axes_manager[0].units}\nMaximum scattering vector: {cutoff_g} {image.axes_manager[0].units}')

    # X, Y = np.meshgrid(image.axes_manager[0].axis, image.axes_manager[1].axis)
    # Set outer cutoff
    # R = np.sqrt(X ** 2 + Y ** 2)
    # cutoff_mask[R >= cutoff_g] = True
    # Set inner cuton
    # R = np.sqrt(X ** 2 + Y ** 2)
    # direct_beam_mask[R <= cuton_k] = True

    # Mask reflections
    # blob_kwargs = {
    #     'min_sigma': 1,
    #     'max_sigma': 15,
    #     'num_sigma': 100,
    #     'overlap': 0,
    #     'threshold': 1.4E1,
    # }
    # # Print some info
    # sep = "\n\t"
    # logger.info(
    #     f'Searching for blobs using arguments:\n\t{f"{sep}".join([f"{key}: {blob_kwargs[key]}" for key in blob_kwargs])}')
    # # Look for blobs (reflections)
    # blobs = blob_log(image.data, **blob_kwargs)
    # logger.info(f'Found {len(blobs)} blobs')
    # # Create masks
    # xs, ys = np.arange(0, nx), np.arange(0, ny)
    # X, Y = np.meshgrid(xs, ys)
    # for blob in blobs:
    #     y, x, r = blob  # x and y axes are flipped in hyperspy compared to numpy
    #     r = np.sqrt(2) * r  # Scale blob radius to appear more like a real radius
    #     r = max([minimum_r, r])  # Make sure that the radius is at least the specified minimum radius
    #     logger.info(f'Adding mask with radius {r} at ({x}, {y})')
    #     R = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
    #     mask[R < r] = True

    # direct_beam_mask = hs.signals.Signal2D(direct_beam_mask)
    # direct_beam_mask.metadata.General.title = f'>{cuton_k} {signal.axes_manager[-1].units} mask'

    # cutoff_mask = hs.signals.Signal2D(cutoff_mask)
    # cutoff_mask.metadata.General.title = f'<{cutoff_g} {signal.axes_manager[-1].units} mask'

    # mask = hs.signals.Signal2D(mask)
    # mask.metadata.General.title = f'Reflection mask'
    # mask.metadata.add_dictionary({'Preprocessing': {'blob_log': blob_kwargs,
    #                                                 'minimum_r': minimum_r}})

    # for m in [direct_beam_mask, mask, cutoff_mask]:
    #     for ax in range(image.axes_manager.signal_dimension):
    #         m.axes_manager[ax].scale = image.axes_manager[ax].scale
    #         m.axes_manager[ax].units = image.axes_manager[ax].units

    # # Add metadata
    # signal.metadata.add_dictionary({
    #     'Preprocessing': {
    #         'Masks': {
    #             'Diffraction': {
    #                 'direct_beam': direct_beam_mask,
    #                 'reflections': mask,
    #                 'cutoff': cutoff_mask
    #             }
    #         }
    #     }
    # })

    # Normalize.
    # Normalize the data to (0, 1.0). This makes the data a float type. Remember to change dtype back to uint 16 if needed!
    logger.info('Normalizing data')
    signal.change_dtype('float32')
    signal = signal / signal.nanmax(axis=[0, 1, 2, 3])

    # Make VBF and maximum through-stack
    logger.info(f'Preparing VBF')
    vbf = signal.get_integrated_intensity(hs.roi.CircleROI(cx=0., cy=0., r_inner=0., r=0.07))
    signal.metadata.add_dictionary({
        'Preprocessing': {'VBF': vbf}
    })

    logger.info('Preparing maximum through-stack')
    maximums = signal.max(axis=[0, 1])
    signal.metadata.add_dictionary({
        'Preprocessing': {'Maximums': maximums}
    })

    # Save the signal
    if isinstance(formats, str):
        formats = [formats]

    for f in formats:
        preprocessed_filename = filename.with_name(f'{filename.stem}_preprocessed{f}')
        logger.info(f'Saving preprocessed data to "{preprocessed_filename.absolute()}"')
        try:
            signal.save(preprocessed_filename, chunks=(32, 32, 32, 32), overwrite=True)
        except Exception as e:
            logger.error(
                f'Exception when saving preprocessed signal with format {f}: \n{e}. \nSkipping format and continuing.')

    # Save the VBF and maximums
    logger.info(f'Saving VBF and maximums as images')
    imsave(filename.with_name(f'{filename.stem}_preprocessed_vbf.png'), vbf.data)
    imsave(filename.with_name(f'{filename.stem}_preprocessed_maximums.png'), maximums.data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=Path, help='Path to a 4D-STEM dataset to preprocess')
    parser.add_argument('-x', '--cx', dest='cx', default=128, type=int, help='Center X position of COM mask')
    parser.add_argument('-y', '--cy', dest='cy', default=128, type=int, help='Center Y position of COM mask')
    parser.add_argument('-r', dest='r', default=12, type=float, help='COM mask radius for descan correction')
    parser.add_argument('-w', '--width', dest='width', default=20, type=int, help='Width of navigation frame to use when estimating descan linear shift')
    parser.add_argument('--calibration', dest='calibration', default=None, help='Diffraction scale given in Å/px')
    parser.add_argument('-l', '--lazy', dest='lazy', action='store_true', help='Work on the data lazily')
    parser.add_argument('-v', '--verbose', dest='verbosity', default=0, action='count', help='Set verbose level')
    parser.add_argument('-f', '--formats', dest='formats', type=list, default=['.hspy'], nargs='?',
                        help='The dataformats to save the preprocessed data to')
    arguments = parser.parse_args()

    log_level = [logging.WARNING, logging.INFO, logging.DEBUG][min([arguments.verbosity, 2])]
    logger.setLevel(log_level)

    preprocess(arguments.filename, lazy=arguments.lazy, com_mask=(arguments.cx, arguments.cy, arguments.r), width=arguments.width, calibration=arguments.calibration, formats=arguments.formats)
