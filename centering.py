import logging
#basic config must be done before loading other packages
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%d-%m-%Y %H:%M:%S')
import argparse
from pathlib import Path
import hyperspy.api as hs
import pyxem as pxm
import numpy as np


def center_4DSTEM(signal, com_mask=(128, 128, 12), beam_shift_estimate_mask=None):
    """
    Center a 4DSTEM signal by estimating linear shifts from center of mass

    :param signal: The signal to be centered
    :param com_mask: The (cx, cy, r) region (in pixels) in diffraction patterns to calculate centre of mass within
    :param beam_shift_estimate_mask: The navigation axes mask to use when estimating linear shifts. True regions will not be used.
    :type signal: pyxem.signals.ElectronDiffraction2D
    :type com_mask: Union[tuple, list]
    :type beam_shift_estimate_mask: hyperspy.signals.Signal2D
    :return: The shifted dataset, with centering parameters added to metadata (Preprocessing.Centering)
    :rtype: pyxem.signals.ElectronDiffraction2D
    """
    if not isinstance(signal, pxm.signals.ElectronDiffraction2D):
        logging.debug(f'Signal {signal!r} did not pass type test')
        logging.warning(f'Signal {signal} should be type `pyxem.signals.ElectronDiffraction2D` not {type(signal)} to perform centering. This might cause problems using this script.')
    else:
        logging.debug(f'Signal {signal!r} passed type test')

    # Calculate COM
    logging.debug('Calculating COM')
    com = signal.center_of_mass(mask=com_mask)

    # Create beam shift object
    logging.debug('Creating BeamShift object')
    beam_shift = pxm.signals.BeamShift(com.T)

    # Estimate beam shift as a linear plane
    if beam_shift_estimate_mask is not None:
        logging.debug(
            f'Estimating linear plane using a mask with {np.count_nonzero(beam_shift_estimate_mask.data)} True values')
    else:
        logging.debug(f'Estimateing linear plane using mask={beam_shift_estimate_mask!r}')
    beam_shift.make_linear_plane(beam_shift_estimate_mask)

    # Apply shift
    logging.debug('Applying beam shift')
    shifted_signal = signal.shift_diffraction(beam_shift.isig[0] - (signal.axes_manager.signal_shape[0] // 2),
                                              beam_shift.isig[1] - (signal.axes_manager.signal_shape[1] // 2))

    # Update metadata
    logging.debug('Creating centering metadata dictionary')
    metadata_dict = {
        'Preprocessing': {
            'Centering': {
                'COM': {
                    'COM': com,
                    'Mask': com_mask},
                'Shifts': {
                    'Shifts': beam_shift,
                    'Mask': beam_shift_estimate_mask}
            }
        }
    }
    logging.debug(f'Updating metadata to {metadata_dict}')
    shifted_signal.metadata.add_dictionary(metadata_dict)
    shifted_signal.original_metadata.add_dictionary(metadata_dict)
    logging.debug(f'Shifted signal metadata:\n{shifted_signal.metadata}')
    return shifted_signal


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', type=str, help='Path to a 4D-STEM dataset to center')
    parser.add_argument('--com_mask', dest='com_mask', default=(128, 128, 12), nargs=3, type=int,
                        help='Mask (cx cy r) to use when calculating center of mass of diffraction patterns')

    beam_shift_estimate_group = parser.add_mutually_exclusive_group()
    beam_shift_estimate_group.add_argument('--beam_shift_estimate_mask', dest='beam_shift_estimate_mask', type=str,
                                           help='Path to a mask to use when estimating a linear shift based on center of mass. Must have same shape as the navigation shape of the signal to be centered.')
    beam_shift_estimate_group.add_argument('--beam_shift_estimate_width', dest='beam_shift_estimate_width', default=20,
                                           type=int,
                                           help='Width of frame around navigation/scan area to use when estimating a linear shift based on center of mass')

    parser.add_argument('-l', '--lazy', dest='lazy', action='store_true', help='Whether to use lazy loading or not')
    parser.add_argument('-o', '--overwrite', dest='overwrite', action='store_true',
                        help='Whether to overwrite the input data or not. If False, a new file with the `[stem]_centered.[suffix]` added to the filename will be created')
    parser.add_argument('-c', '--chunks', dest='chunks', default=None, help='The chunking to use for the output signal')
    parser.add_argument('-v', '--verbose', dest='verbosity', default=0, action='count', help='Set verbose level')

    arguments = parser.parse_args()

    log_level = [logging.WARNING, logging.INFO, logging.DEBUG][min([arguments.verbosity, 2])]
    logging.root.setLevel(log_level)

    args_as_str = [f'\n\t{arg!r} = {getattr(arguments, arg)}' for arg in vars(arguments)]
    logging.debug(f'Running centering script with arguments:{"".join(args_as_str)}')

    # Load signal
    input_name = Path(arguments.filename)
    logging.info(f'Loading "{input_name.absolute()}" with lazy={arguments.lazy}')
    signal = hs.load(input_name, lazy=arguments.lazy)

    # Get the navigation mask
    logging.debug('Getting beam_shift_estimate_mask')
    if arguments.beam_shift_estimate_mask is not None:
        beam_shift_estimate_mask_filename = Path(arguments.beam_shift_estimate_mask)
        logging.info(f'Loading mask for beam shift estimation: "{beam_shift_estimate_mask_filename.absolute()}')
        mask = hs.load(beam_shift_estimate_mask_filename)
    else:
        logging.debug(
            f'I will create a mask for beam shift estimation with shape {signal.axes_manager.navigation_shape}')
        mask = hs.signals.Signal2D(np.zeros(signal.axes_manager.navigation_shape, dtype=bool)).T
        width = abs(arguments.beam_shift_estimate_width)
        logging.info(f'Creating mask for beam shift estimation as a frame of width {width}')
        mask.inav[width:-width, width:-width] = True
        logging.debug(f'Beam shift estimation mask has {np.count_nonzero(mask.data)} True points.')

    # Shift the signal
    logging.info('Centering signal')
    shifted_signal = center_4DSTEM(signal, com_mask=arguments.com_mask, beam_shift_estimate_mask=mask)
    logging.info('Finished centering')

    # Get output name
    if arguments.overwrite:
        output_name = input_name
    else:
        output_name = input_name.with_name(f'{input_name.stem}_centered{input_name.suffix}')
    logging.debug(f'Output name: {output_name.absolute()}')

    # Get chunking
    if arguments.chunks is not None:
        chunks = arguments.chunks
        logging.debug(f'Chunks gotten from parsed input arguments: {chunks}')
    else:
        if arguments.lazy:
            chunks = signal.data.chunksize
            logging.debug(f'Chunks gotten from lazy loaded signal: {chunks}')
        else:
            logging.debug(f'No chunking specified.')
            chunks = None

    # Store output signal
    logging.info(f'Shifted signal will be stored as "{output_name.absolute()}" with chunks={chunks}')
    if chunks is not None:
        shifted_signal.save(output_name, overwrite=True, chunks=chunks)
    else:
        shifted_signal.save(output_name, overwrite=True)

    logging.info('Finished centering script')
