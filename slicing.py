import logging

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', type=str, help='Path to a 4D-STEM dataset to slice')
    parser.add_argument('--navslice', dest='navslice', nargs=6)
    parser.add_argument('--sigzlice', dest='sigslice', nargs=6)

    parser.add_argument('-l', '--lazy', dest='lazy', action='store_true', help='Whether to use lazy loading or not')
    parser.add_argument('-v', '--verbose', dest='verbosity', default=0, action='count', help='Set verbose level')

    arguments = parser.parse_args()

    log_level = [logging.WARNING, logging.INFO, logging.DEBUG][min([arguments.verbosity, 2])]
    logger.setLevel(log_level)

    args_as_str = [f'\n\t{arg!r} = {getattr(arguments, arg)}' for arg in vars(arguments)]
    logger.debug(f'Running slicing script with arguments:{"".join(args_as_str)}')

    if arguments.navslice is None:
        navslice = [None, None, 1, None, None, 1]
        logger.debug(f'No navigation slice specified, using default slicing: {navslice}')
    else:
        navslice = arguments.navslice
        for i, s in enumerate(navslice):
            try:
                navslice[i] = int(s)
            except ValueError as e:
                navslice[i] = None
                logger.error(f'The following error was ignored when parsing navigation slice {s} for axis {i}: {e}')

        logger.debug(f'Userspecified navigation slice: {navslice}')

    if arguments.sigslice is None:
        sigslice = [None, None, 1, None, None, 1]
        logger.debug(f'No signal slice specified, using default slicing: {sigslice}')
    else:
        sigslice = arguments.sigslice
        for i, s in enumerate(sigslice):
            try:
                sigslice[i] = int(s)
            except ValueError as e:
                sigslice[i] = None
                logger.error(f'The following error was ignored when parsing signal slice {s} for axis {i}: {e}')
        logger.debug(f'Userspecified signal slice: {sigslice}')

    # Load signal
    input_name = Path(arguments.filename)
    logger.info(f'Loading "{input_name.absolute()}" with lazy={arguments.lazy}')
    signal = hs.load(input_name, lazy=arguments.lazy)

    # Slice signal
    logger.info(f'Slicing signal with\n\tnavslice: {navslice}\n\tsigslice: {sigslice}')
    sliced_signal = signal.inav[navslice[0]:navslice[1]:navslice[2], navslice[3]:navslice[4]:navslice[5]].isig[sigslice[0]:sigslice[1]:sigslice[2], sigslice[3]:sigslice[4]:sigslice[5]]

    slices_as_string = '_'.join([f'{s}' for s in navslice+sigslice])
    output_name = input_name.with_name(f'{input_name.stem}_sliced_{slices_as_string}{input_name.suffix}')
    logger.info(f'Saving sliced signal to {output_name.absolute()}')
    sliced_signal.save(output_name)

    logger.info('Finished')
