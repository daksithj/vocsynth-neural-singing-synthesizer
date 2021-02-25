import os
import numpy as np
import pyworld
import soundfile
import pysptk
import noisereduce as nr
from pysndfx import AudioEffectsChain
from args import parser

params = parser.parse_args()


def decode_envelopes(spectral_coded, aperiodic_coded, sample_rate, vocal_name):
    # Reverse MFSC to MFCC mirror, remove mirror back. Reduce the scaling of DC and Nynquist frequencies
    # Convert back the MFCC to frequency

    fft_size = params.fft_size
    order = params.mcep_order
    coding_const = params.coding_const
    gamma = params.mcep_gamma
    alpha = params.mcep_alpha

    directory = params.training_dir + '/' + vocal_name + '/'
    [min_spec, max_spec, min_ap, max_ap] = np.load(directory + "min_max.npy", allow_pickle=True)

    spectral_coded = (spectral_coded + coding_const) * (max_spec - min_spec) + min_spec
    mirror = np.fft.irfft(spectral_coded)
    half_mirror = mirror[:, :order]
    half_mirror[:, 0] /= 2
    half_mirror[:, -1] /= 2

    spectral_env = np.exp(np.apply_along_axis(pysptk.mgc2sp, 1, half_mirror, alpha, gamma, fftlen=fft_size).real)

    aperiodic_coded = (aperiodic_coded + coding_const) * (max_ap - min_ap) + min_ap
    aperiodic_coded = np.array(aperiodic_coded, order='C')
    aperiodic_env = pyworld.decode_aperiodicity(aperiodic_coded, sample_rate, fft_size)

    return spectral_env, aperiodic_env


def noise_remover(file_name, reverb):
    audio_data, sample_rate = soundfile.read(file_name)
    noise_part = audio_data[-sample_rate:]
    clean_audio = nr.reduce_noise(audio_data, noise_part, n_grad_freq=params.n_grad_freq,
                                  n_std_thresh=params.n_std_thresh, n_grad_time=params.n_grad_time)

    if reverb is None:
        reverb = params.reverb

    if reverb > 0:
        fx = AudioEffectsChain().reverb(reverberance=reverb)
        clean_audio = fx(clean_audio)

    soundfile.write(file_name, clean_audio, sample_rate)


def construct_audio(spectral_data, aperiodic_data, frequency, file_name, reverb=None):
    output_dir = params.output_dir

    if not os.path.isdir(output_dir + '/'):
        os.mkdir(output_dir)

    audio_data = pyworld.synthesize(frequency, spectral_data, aperiodic_data, params.sample_rate, params.frame_period)

    file_name = output_dir + '/' + file_name + ".wav"
    soundfile.write(file_name, audio_data, params.sample_rate)

    noise_remover(file_name, reverb)
