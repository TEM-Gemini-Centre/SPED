import logging
#basic config must be done before loading other packages
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%d-%m-%Y %H:%M:%S')
import argparse
from pathlib import Path
import hyperspy.api as hs
import pyxem as pxm
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', type=str, help='Path to a 4D-STEM dataset to rebin')
    parser.add_argument('--scale', dest='scale', default=(1, 1, 2, 2), nargs=4, type=int,
                        help='Rebinning scales')
    parser.add_argument('-l', '--lazy', dest='lazy', action='store_true', help='Whether to use lazy loading or not')
    parser.add_argument('-v', '--verbose', dest='verbosity', default=0, action='count', help='Set verbose level')

    arguments = parser.parse_args()

    log_level = [logging.WARNING, logging.INFO, logging.DEBUG][min([arguments.verbosity, 2])]
    logging.root.setLevel(log_level)

    args_as_str = [f'\n\t{arg!r} = {getattr(arguments, arg)}' for arg in vars(arguments)]
    logging.debug(f'Running rebinning script with arguments:{"".join(args_as_str)}')

    # Load signal
    input_name = Path(arguments.filename)
    logging.info(f'Loading "{input_name.absolute()}" with lazy={arguments.lazy}')
    signal = hs.load(input_name, lazy=arguments.lazy)

    #Rebin signal
    rebinned_signal = signal.rebin(scale=arguments.scale)

    scales_as_string = '_'.join([f'{scale:.0f}' for scale in arguments.scale])
    output_name = input_name.with_name(f'{input_name.stem}_rebinned_{scales_as_string}{input_name.suffix}')
    rebinned_signal.save(output_name)
