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
import os
import sys
from os import devnull
from pathlib import Path
import hyperspy.api as hs
import pyxem as pxm
import numpy as np
import pickle
from tabulate import tabulate

from diffsims.libraries.structure_library import StructureLibrary
from diffsims.generators.diffraction_generator import DiffractionGenerator
from diffsims.generators.library_generator import DiffractionLibraryGenerator

from pyxem.utils import indexation_utils as iutls

from pathlib import Path


def load_pickle(filename):
    with open(filename, 'rb') as handle:
        obj = pickle.load(handle)
    return obj



def optimize_parameter(image, library_generator, structure_library, scale, excitation_error, max_radius=None, library_kwargs=None, **matching_kwargs):
    """
    Optimize scale or excitation error for a pattern matching library

    :param image: The diffraction pattern to optimize the library for
    :param library_generator: Library generator to use to generate libraries
    :param structure_library: The structure library to use when generating templates
    :param scale: The scale to perform template matching for. Either `scale` or `excitation_error` should be an iterable, while the other should be a float
    :param excitation_errors: The excitation error to perform template matching for. Either `scale` or `excitation_error` should be an iterable, while the other should be a float
    :param max_radius: The maximum radius in pixels used in template matching
    :param library_kwargs: Keyword arguments passed to the library generator
    :param matching_kwargs: Keyword arguments passed to the matching function
    :type image: pyxem.signals.ElectronDiffraction2D
    :type library_generator: diffsims.generators.library_generator.DiffractionLibraryGenerator
    :type structure_library: diffsims.libraries.structure_library.StructureLibrary
    :type scale: Union[float, array-like]
    :type excitation_error: Union[float, array-like]
    :type max_radius: Union[None, float]
    :returns: optimized_parameter, optimization_signal
    :rtype: tuple
    """
    try:
        n_scale = len(scale)
    except TypeError as e:
        logger.debug(f'Scale has no `len()`:\n{e}\nSetting n_scale=1')
        n_scale = 1

    try:
        n_s = len(excitation_error)
    except TypeError as e:
        logger.debug(f'Excitation_error has no `len()`:\n{e}\nSetting n_s=1')
        n_s = 1

    logger.debug(f'n_scale={n_scale} and n_s={n_s}')
    if n_scale == 1 and n_s == 1:
        raise TypeError(
            f'Optimization routine expects either `scale` or `excitation_error` to be an iterable, not both. I got {type(scale)} and {type(excitation_error)}, respectively, instead')
    elif n_scale > 1 and n_s == 1:
        parameter_name = 'calibration'
        iterable = scale
        constant = excitation_error
    elif n_s > 1 and n_scale == 1:
        parameter_name = 'max_excitation_error'
        iterable = excitation_error
        constant = scale
    else:
        raise TypeError(f'Optimization routine expects either `scale` or `excitation_error` to be an iterable, not both. I got scale of type {type(scale)} and length {n_scale} and excitation errors of type {type(excitation_error)} and length {n_s}.')
    logger.info(f'Iterable parameter is {parameter_name} of length {len(iterable)}')

    n_best = matching_kwargs.get('n_best', 5)
    matching_kwargs['n_best'] = n_best
    table = tabulate([[key, matching_kwargs[key]] for key in matching_kwargs], headers=("Parameter", "Value"))
    logger.debug(f'Keyword arguments for template matching:\n{table}')

    results = np.zeros((len(iterable), 4, n_best))
    half_shape = max(image.axes_manager.signal_shape) // 2
    logger.debug(f'Optimizing library with half shape {half_shape}')

    if library_kwargs is None:
        library_kwargs = {}

    for i, parameter in enumerate(iterable):
        if parameter_name == 'calibration':
            scale = parameter
            excitation_error = constant
        else:
            scale = constant
            excitation_error = parameter
        logger.info(
            f'Simulating pattern {i} of {len(iterable)} with scale {scale} and excitation_error {excitation_error}')

        try:
            if max_radius is None:
                reciprocal_radius = scale * half_shape
            else:
                reciprocal_radius = scale * max_radius

            # Calculate library
            library_kwargs['calibration'] = scale
            library_kwargs['max_excitation_error'] = excitation_error
            library_kwargs['reciprocal_radius'] = reciprocal_radius
            library_kwargs['half_shape'] = half_shape

            # table = tabulate([[key, f'{library_kwargs[key]}'] for key in library_kwargs])
            # logger.debug(f'Calculating library with parameters:\n{table}')

            # Redirect std out due to progressbars clogging up the console output
            with tqdm.external_write_mode():
                diff_lib = library_generator.get_diffraction_library(structure_library, **library_kwargs)

            # Extract simulations
            phase = list(diff_lib.keys())[0]
            simulations = diff_lib[phase]["simulations"]
            logger.info(f'Matching templates for phase {phase} to image')
            # Template match the image with the new library. Output is (indices, angles, correlations, signs), and is stored in the results array

            # Redirect std out due to progressbars clogging up the console output
            with tqdm.external_write_mode():
                results[i, 0, :], results[i, 1, :], results[i, 2, :], results[i, 3, :] = iutls.get_n_best_matches(
                    image.data,
                    simulations,
                    **matching_kwargs)


        except ValueError as e:
            logger.error(
                f'Simulation {i} of {len(iterable)} with scale={scale:.2e} and excitation error {excitation_error:.2e} failed with error {e}. Passing and continuing with next simulations')
        logger.debug(f'Done with pattern matching step {i} of {len(iterable)}')

    # Create a hyperspy signal for easy navigation. A little bit tricky to visualize all of the parameters due to different value scales (correlation scores are usually much lower than 1, while the angles can be from 0 to 360 I think.
    results_signal = hs.signals.Signal2D(results, axes=[
        {'name': parameter_name, 'navigate': False, 'size': results.shape[0], 'scale': iterable[1] - iterable[0],
         'offset': iterable[0]},
        {'name': 'parameter', 'navigate': True,
         'size': results.shape[1]},
        {'name': 'match', 'navigate': False, 'size': results.shape[2]}])
    results_signal.metadata.General.title = f'Template matching {parameter_name} optimization results'
    results_signal.metadata.add_dictionary({'About': {
        'Parameters': {'0': 'Template index', '1': 'In-plane rotation', '2': 'Correlation score', '3': 'Sign'},
    }})

    # Get the optimum parameters
    optimum_parameter = np.where(results[:, 2, 0] == np.max(results[:, 2, 0]))
    optimized_parameter = iterable[optimum_parameter][0]
    logger.info(f'Library optimized for {parameter_name} = {optimized_parameter}')

    return optimized_parameter, results_signal

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', type=Path, help='Path to a 4D-STEM dataset to templatematch')

    parser.add_argument('library', type=Path, help='Path to a pickled structure library')

    parser.add_argument('parameter', type=str, choices=['scale', 's'], help='Which parameter to optimize')

    parameter_group = parser.add_mutually_exclusive_group()
    parameter_group.add_argument('--scale', dest='scale', type=float, default=None, help='Scale to use when matching.')
    parameter_group.add_argument('--max_s', dest='max_s', type=float, default=None, help='Excitation error to use when matching.')

    optimization_group = parser.add_argument_group('Library optimization',
                                                   'Parameters for controlling library optimization. If `parameter` is given as "scale", default value for minimum and maximum values are taken as +/- 10% of signal calibration. if `parameter` is given as "s", default values for minimum and maximum values are taken as 0.01 and 0.1.')
    optimization_group.add_argument('--minimum_value', type=float, help='Minimum value to use for parameter optimization.')
    optimization_group.add_argument('--maximum_value', type=float, help='Maximum value to use for parmeter optimization.')
    optimization_group.add_argument('-n', default=100, type=int,
                                    help='Number of values to use for optimization. default is 100')
    optimization_group.add_argument('--inav', default=[0, 0], type=int, nargs=2,
                                    help='Navigator indices for pattern to use for optiization. Default is 0 0')

    parser.add_argument('-l', '--lazy', dest='lazy', action='store_true', help='Whether to use lazy loading or not')
    parser.add_argument('-v', '--verbose', dest='verbosity', default=0, action='count', help='Set verbose level. Default is 0')

    diffraction_generator_group = parser.add_argument_group('Diffraction generator',
                                                            'Arguments used to create diffraction generator')
    diffraction_generator_group.add_argument('--acceleration_voltage', type=float,
                                             help='Acceleration voltage to use when simulating patterns. If not specified, it will be gotten from the signal metadata')
    diffraction_generator_group.add_argument('--precession_angle', type=float,
                                             help='Precession angle in degrees to use when simulating patterns. If not specified, it will be gotten from the signal metadata')
    diffraction_generator_group.add_argument('--scattering_params', type=str, default='xtables',
                                             help='Scattering parameters to use when simulating patterns. Default is xtables')
    diffraction_generator_group.add_argument('--shape_factor_model', type=str, default='linear',
                                             help='Shape factor model to use when simulating patterns. Default is linear')
    diffraction_generator_group.add_argument('--minimum_intensity', type=float, default=1E-10,
                                             help='Minimum intensity to keep in simulated patterns. Default is 1E-10')
    diffraction_generator_group.add_argument('--with_direct_beam', action='store_true',
                                             help='Whether to create diffraction templates with or without the direct beam.')

    matching_group = parser.add_argument_group('Matching parameters', 'Arguments used in template matching')
    matching_group.add_argument('--n_best', type=int, default=5,
                                help='Number of best solutions to return, in order of descending match. Default is 5')
    keep_group = matching_group.add_mutually_exclusive_group()
    keep_group.add_argument('--frac_keep', default=1.0, type=float,
                            help='Fraction (between 0-1) of templates to do a full matching on. By default all templates will be fully matched (value of 1).')
    keep_group.add_argument('--n_keep', default=None, type=int, help='Number of templates to do a full matching on.')
    matching_group.add_argument('--delta_r', type=float, default=1.0,
                                help='The sampling interval of the radial coordinate in pixels. Default is 1')
    matching_group.add_argument('--delta_theta', type=float, default=1.0,
                                help='The sampling interval of the azimuthal coordinate in degrees. This will determine the maximum accuracy of the in-plane rotation angle. Default is 1 degree')
    matching_group.add_argument('--max_r', type=int, help='Maximum radius to consider in pixel units. By default it is the distance from the center of the patterns to a corner of the image.')
    matching_group.add_argument('--find_direct_beam', action='store_true', help='Find the direct beam in the pattern before matching.')
    matching_group.add_argument('--direct_beam_position', type=int, nargs=2, help='Position of direct beam in the pattern')
    matching_group.add_argument('--normalize_image', action='store_true',
                                help='Normalize the images in the correlation coefficient calculation')
    matching_group.add_argument('--normalize_templates', action='store_true',
                                help='Normalize the templates in the correlation coefficient calculation')
    matching_group.add_argument('--intensity_transform_function', type=str, choices=['log', 'log10', 'gamma'], default=None, help='Intensity transform function to apply to images and templates during matching.')
    matching_group.add_argument('--intensity_offset', type=float, help='Value to be added to the signal before running template matching optimization. Useful if a logarithmic intensity transform is applied to the data')
    matching_group.add_argument('--gamma', type=float, default=0.5, help='Gamma to be used if `gamma` was selected for intensity transform function. Default is 0.5')
    matching_group.add_argument('--custom_intensity_transform_function', type=Path, default=None, help='Path to a python file defining a custom function to apply to both images and templates when performing matching. The file must be named `intensity_transform_function` and contain a function `intensity_transform_function(image)` that takes an image as its only argument')

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

    if arguments.intensity_offset is not None:
        logger.info(f'Adding intensity offset {arguments.intensity_offset} to signal')
        signal = signal + arguments.intensity_offset

    #Get the intensity transform function:
    if arguments.intensity_transform_function is None:
        intensity_transform_function = None
    elif arguments.intensity_transform_function == 'log':
        intensity_transform_function = np.log
    elif arguments.intensity_transform_function == 'log10':
        intensity_transform_function = np.log10
    elif arguments.intensity_transform_function == 'gamma':
        logger.info(f'Using gamma intensity transformation function with gamma {arguments.gamma}')
        intensity_transform_function = lambda x: x**arguments.gamma
    elif arguments.intensity_transform_function == 'custom':
        if arguments.custom_intensity_transform_function is not None:
            raise NotImplementedError('Custom function import is not supported yet.')
            # if arguments.custom_intensity_transform_function.exists():
            #     logger.debug('Attempting to get custom intensity transform function')
            #     try:
            #         logger.debug(f'Adding custom intensity transform function path "{arguments.custom_intensity_transform_function.parent.absolute()}" to pythonpath')
            #         sys.path.insert(1, arguments.custom_intensity_transform_function.parent.absolute())
            #         logger.debug(f'Attempting function import')
            #         import intensity_transform_function
            #         #from .intensity_transform_function import intensity_transform_function
            #         logger.info(f'Loaded custom intensity transform function {intensity_transform_function} successfully')
            #     except Exception as e:
            #         logger.error(f'Could not get custom intensity transform function from "{arguments.custom_intensity_transform_function.absolute()}"')
            #         raise e
            # else:
            #     raise FileExistsError(f'Intensity transform function file {arguments.custom_intensity_transform_function.absolute()} does not exist')
        else:
            intensity_transform_function = None
    else:
        raise ValueError(f'Intensity transform function {arguments.intensity_transform_function} not understood.')
    logger.debug(f'Using intensity transform function {arguments.intensity_transform_function}: {intensity_transform_function}')

    matching_kwargs = {
        'n_best': arguments.n_best,
        'frac_keep': arguments.frac_keep,
        'n_keep': arguments.n_keep,
        'delta_r': arguments.delta_r,
        'delta_theta': arguments.delta_theta,
        'max_r': arguments.max_r,
        'normalize_image': arguments.normalize_image,
        'normalize_templates': arguments.normalize_templates,
        'intensity_transform_function': intensity_transform_function
    }

    # Set up diffraction and library generator
    logger.info(f'Setting up diffraction library generator')
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
    logger.debug(
        f'DiffractionGenerator arguments:\n{tabulate([[key, diff_gen_kwargs[key]] for key in diff_gen_kwargs], headers=("Argument", "Value"))}')
    diff_gen = DiffractionGenerator(**diff_gen_kwargs)
    lib_gen = DiffractionLibraryGenerator(diff_gen)

    # Load template or structure library
    logger.info(f'Loading library from {arguments.library.absolute()}')
    library = load_pickle(arguments.library)
    if not isinstance(library, StructureLibrary):
        raise TypeError(f'Provided library from {arguments.library} must be a StructureLibrary, not {type(library)}')

    if arguments.parameter == 's':
        logger.debug(f'Library will be optimized in terms of maximum excitation error')
        logger.debug(f'Minimum s is {arguments.minimum_value}')
        if arguments.minimum_value is None:
            minimum_value = 0.01
            logger.debug(f'Setting minimum value to default value {minimum_value} for excitation errors')
        else:
            minimum_value = arguments.minimum_value

        if arguments.maximum_value is None:
            maximum_value = 1
            logger.debug(f'Setting maximum value to default value {maximum_value} for excitation errors')
        else:
            maximum_value = arguments.maximum_value

        logger.debug(f'Creating {arguments.n} linearly spaced excitation errors from {minimum_value} to {maximum_value}')
        excitation_error = np.linspace(minimum_value, maximum_value, num=arguments.n)
        if arguments.scale is None:
            scale = signal.axes_manager[-1].scale
        else:
            scale = arguments.scale
    else:
        logger.debug(f'Library will be optimized in terms of calibration scale')
        logger.debug(f'Minimum scale is {arguments.minimum_value}')
        if arguments.minimum_value is None:
            minimum_value = 0.9 * signal.axes_manager[-1].scale
            logger.debug(
                f'Setting minimum scale to 90% of signal scale {signal.axes_manager[-1].scale}: {minimum_value}')
        else:
            minimum_value = arguments.minimum_value

        logger.debug(f'Maximum scale is {arguments.maximum_value}')
        if arguments.maximum_value is None:
            maximum_value = 1.1 * signal.axes_manager[-1].scale
            logger.debug(
                f'Setting minimum scale to 110% of signal scale {signal.axes_manager[-1].scale}: {maximum_value}')
        else:
            maximum_value = arguments.maximum_value
        logger.debug(
            f'Creating {arguments.n} linearly spaced scales from {minimum_value} to {maximum_value}')
        scale = np.linspace(minimum_value, maximum_value, num=arguments.n)
        excitation_error = arguments.max_s

    logger.info(f'Getting image from {signal!r} at inav={arguments.inav} for library optimization')
    image = signal.inav[arguments.inav]

    logger.info(f'Optimizing library')
    optimized_parameter, results_signal = optimize_parameter(image, lib_gen, library, scale, excitation_error, max_radius=arguments.max_r, **matching_kwargs)
    logger.info(f'Finished optimizing library')

    logger.info(f'Optimized {arguments.parameter} = {optimized_parameter:.5e}')

    opt_path = Path(f'{input_name.absolute().parent}/{input_name.stem}_library_optimization_{arguments.n}-{arguments.parameter}.hspy')
    logger.info(f'Saving optimization result to "{opt_path.absolute()}"')
    results_signal.save(opt_path, overwrite=True)
