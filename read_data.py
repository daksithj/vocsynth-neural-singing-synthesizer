import os
import sys
import soundfile
import librosa
import pyworld
import pandas as pd
import numpy as np

from extract_features import extract_timbre_data, extract_phoneme_data
from pre_process_data import process_frequency, process_and_save, match_input_columns

from args import parser

params = parser.parse_args()


# Start processing the data
def pre_process(file_name, training_dir):

    audio_file_name = training_dir + file_name + '.wav'
    lyrics_file_name = training_dir + 'Transcripts/' + file_name + '.txt'

    audio_data, sample_rate = soundfile.read(audio_file_name)
    audio_data = librosa.resample(audio_data, sample_rate, params.sample_rate)
    sample_rate = params.sample_rate

    harvest_frequency, timing = pyworld.harvest(audio_data, sample_rate, f0_floor=params.min_freq,
                                                f0_ceil=params.max_freq, frame_period=params.frame_period)
    frequency = pyworld.stonemask(audio_data, harvest_frequency, timing, sample_rate)
    audio_length = len(frequency)

    phoneme_data = extract_phoneme_data([audio_file_name, lyrics_file_name, audio_length])

    frequency_data = process_frequency(frequency)

    label_data = pd.concat([phoneme_data, frequency_data], axis=1)

    spectral_data, aperiodic_data = extract_timbre_data([audio_data, frequency, timing, sample_rate])

    return [spectral_data, aperiodic_data, label_data, frequency]


def extract_f_labels(frequency, f_data, label_data):

    notes, _, _ = f_data.extract_notes(frequency)

    notes, note_timings = f_data.get_note_data(notes)

    notes = np.expand_dims(notes, axis=0)
    _, note_data = f_data.shift_data(frequency, notes)

    note_data = np.squeeze(note_data, axis=0)

    f_label_data = label_data[:, 256:]
    f_label_data = np.concatenate([note_data, note_timings, f_label_data], axis=1)

    return f_label_data

# Identify the training data
def extract_transcripts(vocal_name):

    training_dir = params.data_dir + '/' + vocal_name + '/'
    index_file_location = training_dir + "index.xlsx"

    transcript_location = training_dir + 'Transcripts/'
    if not os.path.isdir(transcript_location):
        os.mkdir(transcript_location)

    sound_index = pd.read_excel(index_file_location, header=None, index_col=False)

    file_list = []

    for row in sound_index.itertuples():
        lyrics_file_name = transcript_location + str(row[1]) + '.txt'

        text_file = open(lyrics_file_name, "w")
        text_file.write(row[2])
        text_file.close()

        file_list.append(str(row[1]))

    return file_list, training_dir


# Directly load the training data if already processed
def load_training_data(vocal_name):

    directory = params.training_dir + '/' + vocal_name
    if not os.path.isdir(directory):
        sys.exit("The training data folder:" + directory + " does not exist!")

    spectral_data = np.load(directory + "/spectral_data.npy", allow_pickle=True)
    aperiodic_data = np.load(directory + "/aperiodic_data.npy", allow_pickle=True)
    label_data = np.load(directory + "/label_data.npy", allow_pickle=True)
    cutoff_points = np.load(directory + "/cutoff_points.npy", allow_pickle=True)
    frequency = np.load(directory + "/frequency.npy", allow_pickle=True)

    return spectral_data, aperiodic_data, label_data, cutoff_points, frequency


def read_data(vocal_name):
    file_list, training_dir = extract_transcripts(vocal_name)

    spectral_data = pd.DataFrame()
    aperiodic_data = pd.DataFrame()
    label_data = pd.DataFrame()
    cutoff_points = []
    frequency = []

    for file in file_list:
        read_spectral_data, read_aperiodic_data, read_label_data, read_frequency = pre_process(file, training_dir)

        spectral_data = pd.concat([spectral_data, read_spectral_data], axis=0, ignore_index=True)
        aperiodic_data = pd.concat([aperiodic_data, read_aperiodic_data], axis=0, ignore_index=True)
        label_data = pd.concat([label_data, read_label_data], axis=0, ignore_index=True)
        cutoff_points.append(spectral_data.shape[0])

        frequency.extend(read_frequency)

    cutoff_points = np.asarray(cutoff_points)
    frequency = np.asarray(frequency)
    return spectral_data, aperiodic_data, label_data, cutoff_points, frequency


# Read the training data
def read_training_data(vocal_name='', load=False):

    if not load:
        spectral_data, aperiodic_data, label_data, cutoff_points, frequency = read_data(vocal_name)

        data = [spectral_data, aperiodic_data, label_data, cutoff_points, frequency]

        spectral_data, aperiodic_data, label_data, column_list, frequency = process_and_save(data, vocal_name)
    else:
        spectral_data, aperiodic_data, label_data, cutoff_points, frequency = load_training_data(vocal_name)

    return spectral_data, aperiodic_data, label_data, cutoff_points, frequency


def read_test_data(trained_vocal_name, f_data, compare=False):

    vocal_name = "Test"

    spectral_data, aperiodic_data, label_data, cutoff_points, frequency = read_data(vocal_name)

    data = [spectral_data, aperiodic_data, label_data, cutoff_points, frequency]

    spectral_data, aperiodic_data, label_data, column_list, frequency = process_and_save(data, trained_vocal_name,
                                                                                         save=False)

    label_data = match_input_columns(column_list, label_data)

    f_label_data = extract_f_labels(frequency, f_data, label_data)

    if compare:
        return spectral_data, aperiodic_data, label_data, frequency
    else:
        return label_data, f_label_data, frequency


def add_frequency_data(label_data, frequency):

    label_data = label_data[:, 256:]

    frequency_data = process_frequency(frequency)
    frequency_data = np.asarray(frequency_data).astype(np.int)
    label_data = np.concatenate([frequency_data, label_data], axis=1)

    return label_data
