import logging

logger = logging.getLogger(__file__)
logger.propagate = False
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%d-%m-%Y %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)

import hyperspy.api as hs
import pyxem as pxm
import argparse
import time

from pathlib import Path


def decompose(signal, normalize_poissonian_noise=False, algorithm='SVD', output_dimension=None, navmask=None,
              diffmask=None):
    if not isinstance(signal, hs.signals.Signal2D):
        raise TypeError(f'Cannot perform decomposition on {signal!r}. It is not a 2D hyperspy signal')

    if navmask is not None:
        logger.debug(f'Got navigation mask {navmask}')
        if isinstance(navmask, hs.signals.Signal2D):
            logger.debug(f'Navigation mask is a hyperspy signal. Extracting data array')
            navmask = navmask.data
            if not signal.axes_manager.navigtion_shape == navmask.shape:
                logger.warning(
                    f'The navigation mask shape {navmask.shape} does not match signal navigation shape {signal.axes_manager.navigation_shape}')

    if diffmask is not None:
        logger.debug(f'Got diffraction mask {diffmask}')
        if isinstance(diffmask, hs.signals.Signal2D):
            logger.debug(f'Diffraction mask is a hyperspy signal. Extracting data array')
            diffmask = diffmask.data
            if not signal.axes_manager.signal_shape == diffmask.shape:
                logger.warning(
                    f'The diffraction mask shape {navmask.shape} does not match signal diffraction shape {signal.axes_manager.signal_shape}')

    if isinstance(signal, pxm.signals.LazyDiffraction2D) and arguments.algorithm == 'NMF':
        logger.warning(f'Signal {signal} is lazy but specified algorithm {arguments.algorithm} is not compatible with lazy signals.')
        logger.warning(f'I will compute the signal to make it non-lazy and compatible with requested algorithm')
        signal.compute()

    logger.info(f'Starting decomposition')
    tic = time.time()
    signal.decomposition(normalize_poissonian_noise=normalize_poissonian_noise, algorithm=algorithm,
                         output_dimension=output_dimension, navigation_mask=navmask, signal_mask=diffmask)
    toc = time.time()
    logger.info(f'Finished decomposition. Elapsed time: {toc - tic} seconds')


if __name__ == '__main__':
    # Parser arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hs_file', type=Path, help='The HyperSpy .hdf5 file to decompose')
    parser.add_argument('--poissonian', action='store_true',
                        help=r'Whether to account for possionian noise or not. Default is False')
    parser.add_argument('--components', default=None, type=int, help='Output components')
    parser.add_argument('--algorithm', nargs='?', type=str, default='svd',
                        choices=["SVD", "MLPCA", "sklearn_pca", "NMF", "sparse_pca", "mini_batch_sparse_pca", "RPCA",
                                 "ORPCA", "ORNMF"], help='Decomposition algorithm')
    parser.add_argument('--output_path', type=Path, default=None,
                        help=r'The path to store output. Default is the same directory as the input file')
    parser.add_argument('--diffmask', type=Path, default=None, help='Path to diffraction mask data')
    parser.add_argument('--navmask', type=Path, default=None, help='Path to navigation mask data')
    parser.add_argument('--lazy', action='store_true', help='Load lazy or not')
    parser.add_argument('--save_new_signal', action='store_true',
                        help='Save the signal with the decomposition results to a new file. This duplicates the raw data as well')
    parser.add_argument('--save_learning_results', action='store_true', help='Save learning results in separate files')
    parser.add_argument('-v', '--verbose', dest='verbosity', default=0, action='count', help='Set verbose level')
    parser.add_argument('--no_overwrite', action='store_false',
                        help='Whether to not overwrite existing signals')

    arguments = parser.parse_args()

    log_level = [logging.WARNING, logging.INFO, logging.DEBUG][min([arguments.verbosity, 2])]
    logger.setLevel(log_level)

    args_as_str = [f'\n\t{arg!r} = {getattr(arguments, arg)}' for arg in vars(arguments)]
    logger.debug(f'Running decomposition script with arguments:{"".join(args_as_str)}')

    if arguments.output_path is None:
        output_path = arguments.hs_file.parent
        logger.info(f'No output directory specified, I will put outputs at "{output_path.absolute()}"')
    else:
        output_path = arguments.output_path
        if not output_path.exists():
            logger.info(
                f'Specified output path "{output_path.absolute()}" does not exists. I will attempt to create it')
            if not output_path.is_dir():
                raise ValueError(f'Output path "{output_path.absolute()} is not a directory and cannot be created.')
            else:
                logger.info(f'Creating output directory "{output_path.absolute()}"')
                output_path.mkdir(parents=True, exist_ok=True)
                logger.info(f'Created output directory successfully.')
        logger.info(f'I will put outputs at "{output_path.absolute()}"')

    if arguments.diffmask is not None:
        logger.info(f'Loading diffraction mask signal "{arguments.diffmask.absolute()}')
        diffmask = hs.load(arguments.diffmask.absolute())
    else:
        diffmask = None

    if arguments.navmask is not None:
        logger.info(f'Loading navigation mask signal "{arguments.navmask.absolute()}')
        navmask = hs.load(arguments.navmask.absolute())
    else:
        navmask = None

    logger.info(f'Loading data signal "{arguments.hs_file.absolute()}')
    signal = hs.load(arguments.hs_file.absolute(), lazy=arguments.lazy)

    if not signal.data.dtype == 'float32':
        logger.info(f'Changing datatype from {signal.data.dtype} to float32')
        signal.change_dtype('float32')

    decompose(signal, normalize_poissonian_noise=arguments.poissonian, algorithm=arguments.algorithm,
              output_dimension=arguments.components, navmask=navmask, diffmask=diffmask)

    suffix = ''
    if arguments.poissonian:
        suffix += '_poissonian'
    if diffmask is not None:
        suffix += '_diffmask'
    if navmask is not None:
        suffix += '_navmask'

    output_name = arguments.output_path / f'{arguments.hs_file.stem}_{arguments.algorithm}_{arguments.output_components}{suffix}{arguments.hs_file.suffix}'

    if arguments.save_new_signal:
        logger.info(f'Saving decomposed signal to "{output_name}"')
        signal.save(output_name.absolute(), overwrite=arguments.no_overwrite)
    else:
        logger.info(f'Overwriting datafile with decomposed signal (saving to "{arguments.hs_file.absolute()}")')
        signal.save(arguments.hs_file.absolute(), overwrite=True)

    if arguments.save_learning_results:
        logger.info(f'Saving learning results to "{output_name}" (with _loadings and _factors name identifiers)')
        loadings = signal.get_decomposition_loadings()
        factors = signal.get_decomposition_factors()

        loadings.save(output_name.with_name(f'{output_name.stem}_loadings{output_name.suffix}'), overwrite=True)
        factors.save(output_name.with_name(f'{output_name.stem}_factors{output_name.suffix}'), overwrite=True)

    logger.info(f'Finished')
