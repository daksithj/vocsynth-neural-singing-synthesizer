import os
import numpy as np
import pandas as pd
import pickle as pkl

from args import parser
params = parser.parse_args()


def process_frequency(frequency):
    # Mel frequency is linear up to a certain amount and then it is logarithmic
    mel_f = params.mel_c * np.log(1 + frequency / params.corner_freq)
    mel_min = params.mel_c * np.log(1 + params.min_freq / params.corner_freq)
    mel_max = params.mel_c * np.log(1 + params.max_freq / params.corner_freq)

    mel_f[mel_f > 0] = (mel_f[mel_f > 0] - mel_min) * (params.f_bin - 2) / (mel_max - mel_min) + 1
    mel_f[mel_f < 0] = 1
    mel_f[mel_f > params.f_bin - 1] = params.f_bin - 1

    freq_coded = np.rint(mel_f).astype(np.int)
    freq_array = np.zeros((len(frequency), params.f_bin))

    # Code the data (a column for each frequency bin)
    for freq, row in zip(freq_coded, freq_array):
        row[freq] = 1

    frequency_data = pd.DataFrame(freq_array, dtype=int)
    frequency_data = frequency_data.add_prefix('Frequency_')

    return frequency_data


def categorize_data(label_data):

    label_data = pd.get_dummies(data=label_data, columns=['Phoneme'], prefix='Phoneme')
    label_data = pd.get_dummies(data=label_data, columns=['Pre_phoneme'], prefix='Pre_phoneme')
    label_data = pd.get_dummies(data=label_data, columns=['Post_phoneme'], prefix='Post_phoneme')
    label_data = pd.get_dummies(data=label_data, columns=['Phoneme_timings'], prefix='Phoneme_timings')
    column_list = list(label_data.columns.values)

    return label_data, column_list


def code_data(spectral_data, aperiodic_data, directory):

    [min_spec, max_spec, min_ap, max_ap] = np.load(directory + "min_max.npy", allow_pickle=True)
    spectral_data = (spectral_data - min_spec) / (max_spec - min_spec) - params.coding_const
    aperiodic_data = (aperiodic_data - min_ap) / (max_ap - min_ap) - params.coding_const

    return spectral_data, aperiodic_data


def process_and_save(data, vocal_name, save=True):

    spectral_data, aperiodic_data, label_data, cutoff_points, frequency = data

    directory = params.training_dir + '/' + vocal_name

    if not os.path.isdir(directory):
        os.mkdir(directory)

    directory += '/'

    label_data, column_list = categorize_data(label_data)

    if save:
        label_data = np.asarray(label_data).astype(np.int)
        np.save(directory + "label_data.npy", label_data, allow_pickle=True)
        with open(directory + 'column_list.pkl', 'wb') as f:
            pkl.dump(column_list, f)
    else:
        with open(directory + 'column_list.pkl', 'rb') as f:
            column_list = pkl.load(f)

    spectral_data = np.array(spectral_data,  order='C')
    aperiodic_data = np.array(aperiodic_data,  order='C')

    if save:
        min_spec = np.min(spectral_data)
        max_spec = np.max(spectral_data)
        min_ap = np.min(aperiodic_data)
        max_ap = np.max(aperiodic_data)

        min_max = [min_spec, max_spec, min_ap, max_ap]
        np.save(directory + "min_max.npy", min_max, allow_pickle=True)

    spectral_data, aperiodic_data = code_data(spectral_data, aperiodic_data, directory)
    if save:
        np.save(directory + "spectral_data.npy", spectral_data, allow_pickle=True)
        np.save(directory + "aperiodic_data.npy", aperiodic_data, allow_pickle=True)
        np.save(directory + "frequency.npy", frequency, allow_pickle=True)
        np.save(directory + "cutoff_points.npy", cutoff_points, allow_pickle=True)

    return spectral_data, aperiodic_data, label_data, column_list, frequency


def match_input_columns(column_list, test_label_data):

    label_data = pd.DataFrame(columns=column_list)

    for column in column_list:
        try:
            label_data[column] = test_label_data[column]
        except KeyError:
            label_data[column] = 0

    label_data = np.asarray(label_data).astype(np.int)

    return label_data
