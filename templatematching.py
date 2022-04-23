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
from tqdm import tqdm
from contextlib import redirect_stdout, redirect_stderr
from os import devnull
from pathlib import Path
import hyperspy.api as hs
import pyxem as pxm
import numpy as np
import pickle
from tabulate import tabulate

from orix import plot, sampling
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Orientation, Rotation, symmetry
from orix.vector import Vector3d, Miller
from orix.io import load, save

import diffpy
from diffpy.structure import Atom, Structure, Lattice

from diffsims.generators.rotation_list_generators import get_beam_directions_grid
from diffsims.libraries.structure_library import StructureLibrary
from diffsims.libraries.diffraction_library import DiffractionLibrary
from diffsims.generators.diffraction_generator import DiffractionGenerator
from diffsims.generators.library_generator import DiffractionLibraryGenerator

from pyxem.utils import indexation_utils as iutls
from pyxem.utils import plotting_utils as putls
from pyxem.utils import polar_transform_utils as ptutls
from pyxem.utils import expt_utils as eutls

from pathlib import Path


def load_pickle(filename):
    with open(filename, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def reduce_beam_directions(euler_angles, zone_axis, threshold, symmetry, verify=True):
    """
    Reduce a list of euler angles describing beam directions to a subset around a given zone axis.

    :param euler_angles: The list of euler angles in degrees to reduce
    :param zone_axis: A zone axis ([h, k, l]) to focus on
    :param threshold: The allowed angular spread around the zone axis, in degrees.
    :param symmetry: The symmetry of the crystal
    :param plot: Whether to plot results or not. Default is False.
    :type euler_angles: list
    :type zone_axis: array-like
    :type threshold: float
    :type symmetry: orix.Symmetry
    :type plot: bool
    :returns: reduced_euler_angles
    :rtype: list
    """

    zone_axis = Vector3d(zone_axis).in_fundamental_sector(symmetry=symmetry)
    rotations = Rotation.from_euler(np.deg2rad(euler_angles))
    orientations = Orientation(rotations, symmetry=symmetry)
    directions = (orientations * Vector3d([0, 0, 1])).in_fundamental_sector(symmetry)
    misorientations = zone_axis.angle_with(directions)
    reduced_directions = directions[misorientations <= np.deg2rad(threshold)]
    reduced_orientations = orientations[misorientations <= np.deg2rad(threshold)]
    reduced_rotations = rotations[misorientations <= np.deg2rad(threshold)]
    reduced_euler_angles = euler_angles[misorientations <= np.deg2rad(threshold)]

    # Print some info
    logger.info(
        f'Reduced number of orientations from {len(euler_angles)} to {len(reduced_euler_angles)} by restricting to {threshold:.2f} degrees about {zone_axis.data} for symmetry {symmetry.name}')

    if verify:
        deviation = np.abs(np.rad2deg(reduced_orientations.to_euler()) - reduced_euler_angles).flatten()
        deviation.sort()

        if any(deviation >= 0.1):
            logger.debug(
                f'There are {np.count_nonzero(deviation >= 0.1)} "new" reduced orientations with a deviation greater than 0.1 from the rotations extracted from the list of original euler angles')
            non_zero_deviations = deviation[deviation >= 0.1]
            deviation_string = '\n\t'.join([f'{deviation:.2e} deg' for deviation in non_zero_deviations])
            logger.debug(f'The nonzero deviations are:\n\t{deviation_string}')

    return reduced_euler_angles


def simulations_to_signal(simulations, pattern_size, scale):
    """
    Create a signal from simulations

    :param simulations: The simulated library
    :param pattern_size: The size of the diffraction patterns to create
    :param scale: The scale of the diffraction patterns
    :return:
    """
    simulated_patterns = np.zeros((len(simulations),) + (pattern_size, pattern_size))
    for i, simulation in enumerate(simulations):
        simulated_patterns[i, :, :] = simulation.get_diffraction_pattern(pattern_size, sigma=1)
    simulated_patterns = hs.signals.Signal2D(simulated_patterns)
    simulated_patterns.axes_manager[0].name = 'Simulation'
    simulated_patterns.axes_manager[1].name = 'kx'
    simulated_patterns.axes_manager[2].name = 'ky'
    simulated_patterns.axes_manager[1].scale = scale
    simulated_patterns.axes_manager[2].scale = scale
    simulated_patterns.axes_manager[1].units = '$Å^{-1}$'
    simulated_patterns.axes_manager[2].units = '$Å^{-1}$'
    return simulated_patterns


def template_matching(signal, library, symmetries=None, show_progressbar=False, **kwargs):
    """
    Perform template matching on a signal
    :param signal: The signal to templatematch
    :param library: The library to match with
    :param symmetries: Dictionary with symmetries ({'label': orix.quaternion.symmetry}) for the different phases
    :param show_progressbar: Whether to show progressbar during matching or not.
    :param kwargs: Keyword arguments passed to pyxem.utils.indexaton_utils.index_dataset_with_template_rotation
    :return: crystal map with matching results
    :type signal: pyxem.signals.ElectronDiffraction2D
    :type library: diffsims.libraries.diffraction_library.DiffractionLibrary
    :type symmetries: Union[None, dict]
    :type show_progressbar: bool
    """
    if not isinstance(signal, pxm.signals.ElectronDiffraction2D):
        logger.debug(f'Signal {signal!r} did not pass type test')
        logger.warning(
            f'Signal {signal} should be type `pyxem.signals.ElectronDiffraction2D` not {type(signal)} to perform template matching. This might cause problems using this script.')
    else:
        logger.debug(f'Signal {signal!r} passed type test')

    table = tabulate([[key, kwargs[key]] for key in kwargs], headers=['Parameter', 'Value'])
    logger.info(f'Matching {signal!r} with kwargs:\n{table}')

    # Redirect stdout due to progressbars clogging up the console output
    if show_progressbar:
        result, phasedict = iutls.index_dataset_with_template_rotation(signal, library, **kwargs)
    else:
        with open(devnull, 'w') as f:
            with redirect_stdout(f):
                result, phasedict = iutls.index_dataset_with_template_rotation(signal, library, **kwargs)

    logger.info('Creating crystal map from results')
    xmap = iutls.results_dict_to_crystal_map(result, phasedict, diffraction_library=library)
    if symmetries is not None:
        logger.debug('Setting symmetries of crystal map')
        for label in symmetries:
            logger.debug(f'Setting symmetry for phase {label}')
            xmap.phases[label].point_group = symmetries[label]
    return xmap


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', type=Path, help='Path to a 4D-STEM dataset to templatematch')
    parser.add_argument('library', type=Path, help='Path to a pickled template library or structure library')

    parser.add_argument('--scale', dest='scale', type=float, default=None,
                        help='Scale to use when creating library if a stucture library is provided rather than a diffraction library.')
    parser.add_argument('--max_s', dest='max_s', type=float, default=None,
                        help='Maximum excitation error to use when creating library if a stucture library is provided rather than a diffraction library')
    parser.add_argument('-l', '--lazy', dest='lazy', action='store_true', help='Whether to use lazy loading or not')
    parser.add_argument('-v', '--verbose', dest='verbosity', default=0, action='count', help='Set verbose level')

    diffraction_generator_group = parser.add_argument_group('Diffraction generator',
                                                            'Arguments used to create diffraction generator')
    diffraction_generator_group.add_argument('--acceleration_voltage', type=float,
                                             help='Acceleration voltage to use when simulating patterns. If not specified, it will be gotten from the signal metadata')
    diffraction_generator_group.add_argument('--precession_angle', type=float,
                                             help='Precession angle in degrees to use when simulating patterns. If not specified, it will be gotten from the signal metadata')
    diffraction_generator_group.add_argument('--scattering_params', type=str, default='xtables',
                                             help='Scattering parameters to use when simulating patterns.')
    diffraction_generator_group.add_argument('--shape_factor_model', type=str, default='linear',
                                             help='Shape factor model to use when simulating patterns.')
    diffraction_generator_group.add_argument('--minimum_intensity', type=float, default=1E-10,
                                             help='Minimum intensity to keep in simulated patterns.')
    diffraction_generator_group.add_argument('--with_direct_beam', action='store_true',
                                             help='Whether to create diffraction templates with or without the direct beam.')

    matching_group = parser.add_argument_group('Matching parameters', 'Arguments used in template matching')
    matching_group.add_argument('--phases', type=str, nargs='+',
                                help='Names of phases in the library to do an indexation for. By default this is all phases in the library.')
    matching_group.add_argument('--n_best', type=int, default=5,
                                help='Number of best solutions to return, in order of descending match.')
    keep_group = matching_group.add_mutually_exclusive_group()
    keep_group.add_argument('--frac_keep', default=1.0, type=float,
                            help='Fraction (between 0-1) of templates to do a full matching on. By default all templates will be fully matched. See notes for details')
    keep_group.add_argument('--n_keep', default=None, type=int, help='Number of templates to do a full matching on.')
    matching_group.add_argument('--delta_r', type=float, default=1.0,
                                help='The sampling interval of the radial coordinate in pixels.')
    matching_group.add_argument('--delta_theta', type=float, default=1.0,
                                help='The sampling interval of the azimuthal coordinate in degrees. This will determine the maximum accuracy of the in-plane rotation angle.')
    matching_group.add_argument('--max_r', type=int, default=128,
                                help='Maximum radius to consider in pixel units. By default it is the distance from the center of the patterns to a corner of the image.')
    matching_group.add_argument('--normalize_images', action='store_true',
                                help='Normalize the images in the correlation coefficient calculation')
    matching_group.add_argument('--normalize_templates', action='store_true',
                                help='Normalize the templates in the correlation coefficient calculation')
    matching_group.add_argument('--chunks', type=int, nargs=4, default=None,
                                help='Internally the work is done on dask arrays and this parameter determines the chunking of the original dataset. If set to None then no re-chunking will happen if the dataset was loaded lazily. If set to "auto" then dask attempts to find the optimal chunk size.')
    matching_group.add_argument('--parallel_workers', type=int, default=None,
                                help='The number of workers to use in parallel. If set to "auto", the number of physical cores will be used when using the CPU. For GPU calculations the workers is determined based on the VRAM capacity, but it is probably better to choose a lower number.')
    matching_group.add_argument('--target', type=str, default='cpu', choices=("cpu", "gpu"),
                                help='Use "cpu" or "gpu". If "gpu" is selected, the majority of the calculation intensive work will be performed on the CUDA enabled GPU. Fails if no such hardware is available.')
    matching_group.add_argument('--scheduler', type=str, default='threads',
                                help='The scheduler used by dask to compute the result. "processes" is not recommended.')
    matching_group.add_argument('--precision', type=str, default='float32',
                                help='The level of precision to work with on internal calculations')
    matching_group.add_argument('--show_matching_progressbar', action='store_true',
                                help='SHow the progressbar from template matching algorithm during matching. This might clog up output when run on a cluster')
    matching_group.add_argument('--intensity_transform_function', type=str, choices=['log', 'log10', 'gamma'],
                                default=None,
                                help='Intensity transform function to apply to images and templates during matching.')
    matching_group.add_argument('--intensity_offset', type=float,
                                help='Value to be added to the signal before running template matching optimization. Useful if a logarithmic intensity transform is applied to the data')
    matching_group.add_argument('--gamma', type=float, default=0.5,
                                help='Gamma to be used if `gamma` was selected for intensity transform function. Default is 0.5')

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

    # Set up matching kwargs
    if arguments.chunks is None:
        chunks = 'auto'
    else:
        chunks = arguments.chunks

    if arguments.parallel_workers is None:
        parallel_workers = 'auto'
    else:
        parallel_workers = arguments.parallel_workers

    # Get the intensity transform function:
    if arguments.intensity_transform_function is None:
        intensity_transform_function = None
    elif arguments.intensity_transform_function == 'log':
        intensity_transform_function = np.log
    elif arguments.intensity_transform_function == 'log10':
        intensity_transform_function = np.log10
    elif arguments.intensity_transform_function == 'gamma':
        logger.info(f'Using gamma intensity transformation function with gamma {arguments.gamma}')

        def gamma(image, gamma_value=arguments.gamma):
            return image**gamma_value

        intensity_transform_function = gamma
    else:

        raise ValueError(f'Intensity transform function {arguments.intensity_transform_function} not understood.')
    logger.debug(f'Using intensity transform function {arguments.intensity_transform_function}: {intensity_transform_function}')

    matching_kwargs = {
        'phases': arguments.phases,
        'n_best': arguments.n_best,
        'frac_keep': arguments.frac_keep,
        'n_keep': arguments.n_keep,
        'delta_r': arguments.delta_r,
        'delta_theta': arguments.delta_theta,
        'max_r': arguments.max_r,
        'normalize_images': arguments.normalize_images,
        'normalize_templates': arguments.normalize_templates,
        'chunks': chunks,
        'parallel_workers': parallel_workers,
        'target': arguments.target,
        'scheduler': arguments.scheduler,
        'precision': arguments.precision,
        'intensity_transform_function': intensity_transform_function
    }

    # Load template or structure library
    logger.info(f'Loading library from {arguments.library.absolute()}')
    library = load_pickle(arguments.library)
    logger.info(f'Library is type {type(library)}')
    if isinstance(library, DiffractionLibrary):
        diffraction_library = library
        structure_library = None
    elif isinstance(library, StructureLibrary):
        structure_library = library
        diffraction_library = None
    else:
        raise TypeError(
            f'Library {library!r} of type {type(library)} not understood. It should either be a DiffractionLibrary or a StructureLibrary')
    logger.info(
        f'Using libraries:\n{tabulate([["Diffraction", type(diffraction_library)], ["Structure", type(structure_library)]], headers=("Library", "Type"))}')

    if diffraction_library is None:
        logger.info(f'Creating diffraction library...')

        # Set up diffraction and library generator
        logger.debug(f'Setting up diffraction library generator')
        diff_gen_kwargs = {}
        if arguments.acceleration_voltage is None:
            diff_gen_kwargs['accelerating_voltage'] = signal.metadata.Acquisition_instrument.TEM.beam_energy
        else:
            diff_gen_kwargs['accelerating_voltage'] = arguments.acceleration_voltage
        if arguments.precession_angle is None:
            diff_gen_kwargs['precession_angle'] = signal.metadata.Acquisition_instrument.TEM.rocking_angle
        else:
            diff_gen_kwargs['precession_angle'] = arguments.precession_angle
        diff_gen_kwargs['scattering_params'] = arguments.scattering_params
        diff_gen_kwargs['shape_factor_model'] = arguments.shape_factor_model
        diff_gen_kwargs['minimum_intensity'] = arguments.minimum_intensity
        logger.info(
            f'DiffractionGenerator arguments:\n{tabulate([[key, diff_gen_kwargs[key]] for key in diff_gen_kwargs], headers=("Argument", "Value"))}')
        diff_gen = DiffractionGenerator(**diff_gen_kwargs)
        lib_gen = DiffractionLibraryGenerator(diff_gen)

        logger.debug(
            f'Using provided scale ({arguments.scale}) and excitation error ({arguments.max_s}) for creating diffraction library')
        if arguments.scale is None:
            scale = signal.axes_manager[-1].scale
            logger.debug(
                f'Supplied scale is {arguments.scale}. Using signal calibration {scale} as scale for templates')
        else:
            scale = arguments.scale
        logger.debug(f'Using structure library to create template library')
        diffraction_library = lib_gen.get_diffraction_library(structure_library,
                                                              calibration=scale,
                                                              reciprocal_radius=arguments.max_r * scale,
                                                              half_shape=signal.axes_manager.signal_shape[0] // 2,
                                                              with_direct_beam=arguments.with_direct_beam,
                                                              max_excitation_error=arguments.max_s)

    logger.info('Starting template matching...')
    xmap = template_matching(signal, diffraction_library, show_progressbar=arguments.show_matching_progressbar,
                             **matching_kwargs)
    logger.info('Finished template matching...')

    logger.debug(f'Template matching results:\n{xmap}')

    xmap_path = input_name.with_name(f'{input_name.stem}_crystalmap.hdf5')
    logger.info(f'Saving crystal map to "{xmap_path.absolute()}"')
    save(xmap_path, xmap, overwrite=True)
