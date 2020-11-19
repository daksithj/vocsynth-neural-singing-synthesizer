import pysptk
import pyworld
import copy
import pandas as pd
import numpy as np
from p2fa import align

from args import parser
params = parser.parse_args()


def extract_phoneme_data(args):
    audio_file_name, lyrics_file_name, audio_length = args
    print(audio_file_name)

    # Extract phonemes using Penn's force aligner
    ph_align, w_align = align.align(audio_file_name, lyrics_file_name)

    step = params.frame_period / 1000
    phoneme_position = 0
    phoneme_array = []
    x = 0

    while x < audio_length:
        if ph_align[phoneme_position][1] <= x * step:
            if ph_align[phoneme_position][2] > x * step:
                phoneme_array.append(ph_align[phoneme_position][0])
                x = x + 1
            elif phoneme_position + 1 < len(ph_align):
                phoneme_position = phoneme_position + 1
            else:
                x = x + 1
                phoneme_array.append(ph_align[phoneme_position][0])
        else:
            phoneme_array.append("sp")
            x = x + 1

    x = 0
    timing_array = []

    while x < audio_length:
        phoneme = phoneme_array[x]
        counter = 0
        while x + counter < audio_length and phoneme_array[x + counter] == phoneme:
            counter = counter + 1
        numerator = 1
        for y in range(counter):
            # timing_array.append(numerator / counter)
            timing = numerator / counter
            if timing <= 0.333:
                timing_array.append(0)
            elif timing <= 0.666:
                timing_array.append(1)
            else:
                timing_array.append(2)
            numerator = numerator + 1
        x = x + counter

    pre_phoneme_array = []
    post_phoneme_array = []

    for y in range(audio_length):
        try:
            pre_phoneme_array.append(phoneme_array[y-1])
        except IndexError:
            pre_phoneme_array.append("sp")
        try:
            post_phoneme_array.append(phoneme_array[y+1])
        except IndexError:
            post_phoneme_array.append("sp")

    phoneme_position_data = pd.DataFrame(phoneme_array, columns=['Phoneme'])
    pre_phoneme_position_data = pd.DataFrame(pre_phoneme_array, columns=['Pre_phoneme'])
    post_phoneme_position_data = pd.DataFrame(post_phoneme_array, columns=['Post_phoneme'])
    phoneme_timing_data = pd.DataFrame(timing_array, columns=['Phoneme_timings'])
    phoneme_data = pd.concat([phoneme_position_data, phoneme_timing_data, pre_phoneme_position_data,
                              post_phoneme_position_data], axis=1)
    return phoneme_data


def extract_timbre_data(args):
    audio_data, frequency, timing, sample_rate = args
    # Spectral envelope is taking the frequency-time of the audio and taking short time windows (frames)
    # and Fourier transforming them, to convert to the frequency domain
    spectral_data = pyworld.cheaptrick(audio_data, frequency, timing, sample_rate)
    aperiodic_data = pyworld.d4c(audio_data, frequency, timing, sample_rate)

    # 1. First take spectral envelope and convert it to mel cepstrum (MFCC)
    #     1.1 Spectral envelope is the Short time fourier transform of the frequencies to freuqency bins
    #     1.2 In MFCC we first map the powers of the spectrum to the mel scale
    #     1.3 Take the logs of each mel frequency and take the Discrete Cosine Transform to get MFCC
    #     1.4 MFCC are in the form of amplitudes. The bands used are in the range of what humans can distinct
    #     rather than normal ranges (in normal spec env each band human cant distictively identify)
    # 2. After breaking down into bins in MFCC the DC frequency (at bin 0) and Nyquist (last frequency) scaled by two
    # 3. Using the above a mirror spectrum is created
    # 4. The fourier transform is taken to get the MFSC. MFCC -> Discrete conside transform -> MFSC.
    # Reverse done here to get real values in frequency range

    mcep_floor = 10 ** (-80 / 20)

    spectral_mel = np.apply_along_axis(pysptk.mcep, 1, spectral_data, params.mcep_order - 1,
                                       params.mcep_alpha, itype=params.mcep_input, threshold=mcep_floor)
    scale_mel = copy.copy(spectral_mel)
    scale_mel[:, 0] *= 2
    scale_mel[:, -1] *= 2
    # Create mirror. scale_mel[:, -1:0:-1]] -> all rows, from last column to first,
    # in reverse (the last -1 in the notation)
    mirror = np.hstack([scale_mel[:, :-1],  scale_mel[:, -1:0:-1]])
    mfsc = np.fft.rfft(mirror).real
    spectral_data = pd.DataFrame(mfsc)

    aperiodic_data = pyworld.code_aperiodicity(aperiodic_data, sample_rate)
    aperiodic_data = pd.DataFrame(aperiodic_data)

    return spectral_data, aperiodic_data
