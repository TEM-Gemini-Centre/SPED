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
import pyxem as pxm
import numpy as np
from tabulate import tabulate
from skimage.io import imsave

calibrations = {
    '2100F':
        {
            'Merlin': {
                200: {
                    'CL': {
                        8: {'CL': 16.07, 'scale': 0.0136},
                        10: {'CL': 19.319, 'scale': 0.011},
                        12: {'CL': 22.586, 'scale': 0.0097},
                        15: {'CL': 27.475, 'scale': 0.008},
                        20: {'CL': 35.509, 'scale': 0.0062},
                        25: {'CL': 43.942, 'scale': 0.0050},
                        30: {'CL': 52.493, 'scale': 0.0042},
                        40: {'CL': 69.429, 'scale': 0.0032},
                        50: {'CL': 86.296, 'scale': 0.0025},
                        60: {'CL': 103.702, 'scale': 0.0021},
                        80: {'CL': 136.743, 'scale': 0.0016}
                    }
                }
            }
        }
}


def wavelength(V, m0=9.1093837015 * 1e-31, e=1.60217662 * 1e-19, h=6.62607004 * 1e-34, c=299792458):
    """
    Return the wavelength of an accelerated electron in [Å]

    Arguments
    ---------
    V : float, Acceleration voltage of electrons [kV]
    m0 : float, Rest mass of electron [kg]
    e : float, Elementary charge of electron [C]
    h : float, Planck' constant [m^2 kg/s]
    c : float, Speed of light in vacuum [m/s]
    """
    V = V * 1E3
    return h / np.sqrt(2 * m0 * e * V * (1.0 + (e * V / (2 * m0 * c ** 2)))) * 1E10


def load_hdr(filename):
    """load a header file"""
    filename = Path(filename)
    hdr_content = dict()
    if filename.exists() and filename.suffix == '.hdr':
        with filename.open('r') as hdrfile:
            lines = hdrfile.readlines()
            for lineno, line in enumerate(lines):
                if 0 < lineno < len(lines) - 1:
                    field, value = line.split(':', maxsplit=1)
                    field = field.strip()
                    value = value.strip()
                    hdr_content[field] = value
    else:
        raise FileNotFoundError(f'HDR file "{filename.absolute()}" does not exist or is not a valid .hdr file.')
    return hdr_content


def convert(filename, nx=None, ny=None, chunks=(32, 32), overwrite=True, dx=None,
            dy=None, format='.hspy', normalize=False, log=False, log_shift=1, vbf=True, vbf_half_width=10, **kwargs):
    """
    Convert a .mib file to .hspy format
    :param filename: The path to the .mib file
    :param nx: scan shape in x-direction. Not required for square scans. For non-square scans, either `nx` or `ny`must be provided.
    :param ny: scan shape in y-direction. Not required for square scans. For non-square scans, either `nx` or `ny`must be provided.
    :param detector_shape: Detector shape
    :param chunks: The chunking to use
    :param dx: Scan step size in nm along x
    :param dy: Scan step size in nm along y
    :param format: The output format to use
    :param normalize: Whether to normalize the data during the conversion process. This will store the data in a float32 format which will increase disk storage dramatically.
    :param log: Whether to take a logarithm of the data during the conversion process as part of the normalization routine. If `normalize` is False, this parameter has no effect.
    :param log_shift: An offset to apply to the data before taking the logarithm. This is useful to avoid NaNs in the data where the raw data is zero.
    :param vbf: Whether to also create a vbf .png image of the data
    :param vbf_half_width: The half-width of the square used to create a cheap VBF in pixels.
    :param kwargs: Additional keyword arguments to be added to the metadata of the signal. Specify parameters similarly to pyxems `set_experimental_parameters`, as well as providing your own (e.g. `notes="this is a note"). If `microscope`, `camera`, `beam_energy` and `camera_length` are all provided, a diffraction scale calibration from a table will also be performed as long as a match can be found.
    :type filename: Union[str, Path]
    :type nx: Union[None, int]
    :type ny: Union[int, int]
    :type detector_shape: tuple
    :type chunks: tuple
    :type dx: float
    :type dy: float
    :type format: str
    :type normalize: bool
    :type log: bool
    :type log_shift: Union[int, float]
    :type vbf: bool
    :type vbf_radius: int
    :return: converted signal
    :rtype: pxm.signals.LazyElectronDiffraction2D
    """

    filename = Path(filename)
    logger.info(f'Loading data "{filename.absolute()}" for conversion')
    #signal = pxm.load_mib(str(filename), reshape=False)
    signal = hs.load(str(filename), navigation_shape=(nx, ny))
    logger.debug(f'Loaded signal {signal}')

    try:
        hdr = load_hdr(filename.with_suffix('.hdr'))
    except FileNotFoundError as e:
        logger.error(e)
        hdr = None
    else:
        logger.debug(f'Read header contents from {filename.with_suffix(".hdr")})')

    if normalize or format == '.blo':
        logger.debug('Normalizing data')
        if log:
            logger.debug(
                f'Taking the logarithm (base 10 with an offset of {log_shift}) of the data before normalization')
            signal = np.log10(signal + log_shift)
        signal = signal / signal.max(axis=[0,1,2,3]).data
        logger.info(f'Normalized data')

    if format == '.blo':
        logger.debug('Rescaling data to 8-bit limits for blockfile conversion')
        signal = signal * 2 ** 8

    #logger.debug(f'Rechunking signal to use {chunks[0]} in the navigation dimension and {chunks[1]} in the signal dimension')
    #signal = signal.rechunk(nav_chunks = chunks[0], sig_chunks=chunks[1])
    #logger.debug(f'Rechunked signal:\n{signal}')

    # Get the scan calibration
    if dx is None and dy is None:
        logger.debug(f'No scan calibration is provided')
        pass
    elif dx is None:
        logger.debug(f'Scan y calibration is provided. Treating it as calibration for both x and y directions')
        signal.set_scan_calibration(dy)
    elif dy is None:
        logger.debug(f'Scan x calibration is provided. Treating it as calibration for both x and y directions')
        signal.set_scan_calibration(dx)
    else:
        logger.debug(f'Scan calibration values for both directions are provided.')
        signal.set_scan_calibration(dx)
        signal.axes_manager['y'].scale = dy

    # Get other metadata:
    signal.set_experimental_parameters(
        beam_energy=kwargs.get('beam_energy', None),
        camera_length=kwargs.get('camera_length', None),
        scan_rotation=kwargs.get('scan_rotation', None),
        convergence_angle=kwargs.get('convergence_angle', None),
        rocking_angle=kwargs.get('rocking_angle', None),
        rocking_frequency=kwargs.get('rocking_frequency', None),
        exposure_time=kwargs.get('exposure_time', None)
    )

    if kwargs.get('camera_length', None) is not None:
        try:
            actual_camera_length = \
                calibrations[kwargs.get('microscope', None)][kwargs.get('camera', None)][
                    kwargs.get('beam_energy', None)][
                    'CL'][kwargs.get('camera_length')]['CL']
            diffraction_scale = \
                calibrations[kwargs.get('microscope', None)][kwargs.get('camera', None)][
                    kwargs.get('beam_energy', None)][
                    'CL'][kwargs.get('camera_length')]['scale']
            logger.debug(
                f'Extracted actual camera length ({actual_camera_length} cm) and diffraction scale ({diffraction_scale} 1/Å) from calibration table')

            signal.set_diffraction_calibration(diffraction_scale)
            logger.debug(f'Set diffraction scale')

            signal.set_experimental_parameters(camera_length=actual_camera_length)
            logger.debug(f'Updated camera_length in experimental parameters')
        except KeyError as e:
            table = tabulate(
                [[key, kwargs.get(key, None)] for key in ['microscope', 'camera', 'beam_energy', 'camera_length']],
                headers=['Required field', 'Value'])
            logger.error(
                f'Could not extract actual cameralength and diffraction scale from calibration table due to missing required metadata fields:\n{table}\nError: {e}\nContinuing without calibrating diffraction scale')
    else:
        logger.debug('Could not set diffraction scale due to no specified cameralength')

    signal.original_metadata.add_dictionary({'Acquisition_instrument': {'TEM': {'Acquisition_parameters': kwargs}}})
    signal.metadata.add_dictionary({'Acquisition_instrument': {'TEM': {'Acquisition_parameters': kwargs}}})
    logger.debug(f'Added kwargs to metadata')

    logger.info(
        f'Converted signal {signal} has axes manager:\n{signal.axes_manager}\nand metadata data:\n{signal.metadata}')

    chunks = chunks + (32, 32)

    output_path = filename.with_name(f'{filename.stem}{format}')
    logger.info(f'Storing converted signal to "{output_path.absolute()}" in {chunks} chunks')
    try:
        signal.save(str(output_path), chunks=chunks, overwrite=overwrite)
    except Exception as e:
        logger.error(f'Could not save signal due to error {e}. Trying again without using specified chunking')
        signal.save(str(output_path), overwrite=overwrite)

    if vbf:
        cx, cy = np.array(signal.axes_manager.signal_shape) // 2  # The center of the diffraction patterns
        logger.info(f'Creating VBF image using a square centered at {cx}x{cy} and half-width {vbf_half_width}')
        vbf = signal.isig[cx-vbf_half_width:cx+vbf_half_width, cy-vbf_half_width:cy+vbf_half_width].sum(axis=(2, 3))
        if isinstance(vbf, pxm.signals.LazyDiffraction2D):
            vbf.compute()
        imdata = vbf.data
        imdata = (imdata/np.max(imdata)*2**8).astype(np.uint8)
        logger.info(f'Saving VBF image.')
        imsave(output_path.with_suffix('.png'), imdata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', type=str, help='Path to a 4D-STEM dataset to convert')
    parser.add_argument('-x', '--nx', dest='nx', type=int,
                        help='Scan shape in x-direction. Not required for square scans. Either x or y shape must be given for non-square scans.')
    parser.add_argument('-y', '--ny', dest='ny', type=int,
                        help='Scan shape in y-direction. Not required for square scans. Either x or y shape must be given for non-square scans.')
    parser.add_argument('--chunks', default=(32, 32), dest='chunks', type=int, nargs=2,
                        help='Chunksize to use in the for the navigation and signal spaces')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing converted data')
    parser.add_argument('--format', dest='format', type=str, default='.hspy', choices=['.hspy', '.hdf5', '.blo'],
                        help='Output fileformat')
    parser.add_argument('--normalize', dest='normalize', action='store_true',
                        help='Normalize data before conversion. For blockfile conversion, this will always be performed and this argument has no effect')
    parser.add_argument('--log', dest='log', action='store_true',
                        help='Take the logarithm (base 10) of the data before normalization. Only has an effect if `--normalize` or `--format .blo` is also specified')
    parser.add_argument('--log_shift', dest='log_shift', type=float,
                        help='Offset to add to the data before doing the logarithm if `--log` or `--format .blo` is given')
    parser.add_argument('--vbf', action='store_true',
                        help='Whether to also create a cheap VBF (using a central square as the virtual aperture) image of the dataset')
    parser.add_argument('--vbf_half_width', dest='vbf_half_width', type=int, default=10,
                        help='The half-width of the square used to create a VBF.')
    parser.add_argument('-v', '--verbose', dest='verbosity', default=0, action='count', help='Set verbose level')

    metadata_group = parser.add_argument_group('Optional metadata',
                                               'Optional metadata to be added to the signal. If `beam_energy`, `microscope`, `camera`, and `camera_length` is provided, a diffraction calibration is performed is a match is found in a calibration table.')
    metadata_group.add_argument('--dx', dest='dx', type=float, help='Scan step size in x-direction')
    metadata_group.add_argument('--dy', dest='dy', type=float, help='Scan step size in y-direction')
    metadata_group.add_argument('--beam_energy', dest='beam_energy', type=float, default=200, help='Beam energy in kV')
    metadata_group.add_argument('--camera_length', dest='camera_length', type=float, help='Nominal camera length in cm')
    metadata_group.add_argument('--rocking_angle', dest='rocking_angle', type=float, help='Rocking angle in degrees')
    metadata_group.add_argument('--rocking_frequency', dest='rocking_frequency', type=float,
                                help='Rocking frequency in Hz')
    metadata_group.add_argument('--exposure_time', dest='exposure_time', type=float, help='Exposure time in ms')
    metadata_group.add_argument('--convergence_angle', dest='convergence_angle', type=float,
                                help='Convergence angle in mrad')
    metadata_group.add_argument('--microscope', dest='microscope', type=str, default='2100F', choices=['2100F'],
                                help='Microscope that the data was acquired on')
    metadata_group.add_argument('--camera', dest='camera', type=str, default='Merlin', choices=['Merlin'],
                                help='Camera used to acquire the data')
    metadata_group.add_argument('--mode', dest='mode', type=str, default='NBD', choices=['NBD', 'STEM', 'LMSTEM'],
                                help='Microscope mode used to acquire the data')
    metadata_group.add_argument('--alpha', dest='alpha', type=int, default=5, choices=[1, 2, 3, 4, 5]
                                , help='Microscope alpha setting used to acquire data')
    metadata_group.add_argument('--spotsize', dest='spotsize', type=float, help='Nominal spotsize in nm')
    metadata_group.add_argument('--operator', dest='operator', type=str, help='Operator')
    metadata_group.add_argument('--specimen', dest='specimen', type=str, help='Specimen')
    metadata_group.add_argument('--notes', dest='notes', type=str, help='Notes to add to the metadata fields')
    metadata_group.add_argument('--scan_rotation', dest='scan_rotation', type=float, help='Scan rotation')
    arguments = parser.parse_args()

    log_level = [logging.WARNING, logging.INFO, logging.DEBUG][min([arguments.verbosity, 2])]
    logging.root.setLevel(log_level)

    args_as_str = [f'\n\t{arg!r} = {getattr(arguments, arg)}' for arg in vars(arguments)]
    logging.debug(f'Running conversion script with arguments:{"".join(args_as_str)}')

    _ = convert(arguments.filename, arguments.nx, arguments.ny, arguments.chunks,
                arguments.overwrite, arguments.dx, arguments.dy, arguments.format, arguments.normalize, arguments.log,
                arguments.log_shift, arguments.vbf, arguments.vbf_half_width, beam_energy=arguments.beam_energy,
                camera_length=arguments.camera_length,
                rocking_angle=arguments.rocking_angle, rocking_frequency=arguments.rocking_frequency,
                exposure_time=arguments.exposure_time, convergence_angle=arguments.convergence_angle,
                microscope=arguments.microscope, camera=arguments.camera, mode=arguments.mode, alpha=arguments.alpha,
                spotsize=arguments.spotsize, operator=arguments.operator, specimen=arguments.specimen,
                scan_rotation=arguments.scan_rotation, notes=arguments.notes)
