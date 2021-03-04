import os
import sys
import soundfile
import librosa
import pyworld
import pandas as pd
import numpy as np
import azapi

from extract_features import extract_timbre_data, extract_phoneme_data
from pre_process_data import process_frequency, process_and_save, match_input_columns
from frequency_tools import extract_notes, notes_to_number, get_note_data, note_to_frequency
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


def extract_f_labels(frequency, f_data, label_data, note_file=None, de_tune=False):

    if note_file is None:
        notes, _, _ = extract_notes(frequency)

        if de_tune:
            notes, frequency = f_data.de_tune(notes, frequency)
    else:
        notes = read_notes(note_file, label_data.shape[0])
        notes = notes_to_number(notes)
        frequency = note_to_frequency(notes)
        notes = np.asarray(notes)
        frequency = np.asarray(frequency)

    notes, note_timings = get_note_data(notes)

    notes = np.expand_dims(notes, axis=0)
    _, note_data = f_data.shift_data(frequency, notes, shift=False)

    note_data = np.squeeze(note_data, axis=0)

    f_label_data = label_data[:, 256:]
    f_label_data = np.concatenate([note_data, note_timings, f_label_data], axis=1)

    return f_label_data, frequency


# Identify the training data
def extract_transcripts(data_dir, index_name="index.xlsx"):

    training_dir = data_dir + '/'
    index_file_location = training_dir + index_name

    transcript_location = training_dir + 'Transcripts/'
    if not os.path.isdir(transcript_location):
        os.mkdir(transcript_location)

    index_file_type = index_name.split(".")[1]

    if index_file_type == "xlsx" or index_file_type == "xls":
        sound_index = pd.read_excel(index_file_location, header=None, index_col=False)
    else:
        sound_index = pd.read_csv(index_file_location, header=None, index_col=False, skip_blank_lines=True)

    file_list = []

    for row in sound_index.itertuples():
        lyrics_file_name = transcript_location + str(row[1]) + '.txt'

        text_file = open(lyrics_file_name, "w")
        if str(row[2]) == '':
            lyrics = extract_lyrics(row[3], row[4])
        else:
            lyrics = row[2]
        text_file.write(lyrics)
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


def read_data(data_dir, index_name="index.xlsx", gui_screen=None):
    file_list, training_dir = extract_transcripts(data_dir, index_name)

    spectral_data = pd.DataFrame()
    aperiodic_data = pd.DataFrame()
    label_data = pd.DataFrame()
    cutoff_points = []
    frequency = []

    file_count = 0
    total_files = len(file_list)

    for file in file_list:

        if gui_screen is not None:
            gui_screen.ids.dataset_progress_file.text = f'Processing file: {file}.wav'

        read_spectral_data, read_aperiodic_data, read_label_data, read_frequency = pre_process(file, training_dir)

        spectral_data = pd.concat([spectral_data, read_spectral_data], axis=0, ignore_index=True)
        aperiodic_data = pd.concat([aperiodic_data, read_aperiodic_data], axis=0, ignore_index=True)
        label_data = pd.concat([label_data, read_label_data], axis=0, ignore_index=True)
        cutoff_points.append(spectral_data.shape[0])

        frequency.extend(read_frequency)
        file_count += 1

        if gui_screen is not None:
            if gui_screen.kill_signal:
                gui_screen.ids.dataset_progress_status.text = "Data processing cancelled!"
                gui_screen.ids.dataset_progress_file.text = "Press 'Finish' to return back to menu"
                gui_screen.ids.dataset_finish_button.disabled = False
                gui_screen.ids.dataset_cancel_button.disabled = True
                sys.exit()
            progress = int((file_count/total_files)*100)
            gui_screen.ids.dataset_progress_bar.value = progress
            gui_screen.ids.dataset_progress_value.text = f'{progress}% Complete'

    cutoff_points = np.asarray(cutoff_points)
    frequency = np.asarray(frequency)
    return spectral_data, aperiodic_data, label_data, cutoff_points, frequency


# Read the training data
def read_training_data(data_dir, vocal_name='', index_name="index.xlsx", gui_screen=None, load=False):

    if not load:
        spectral_data, aperiodic_data, label_data, cutoff_points, frequency = read_data(data_dir, index_name,
                                                                                        gui_screen)

        data = [spectral_data, aperiodic_data, label_data, cutoff_points, frequency]

        spectral_data, aperiodic_data, label_data, column_list, frequency = process_and_save(data, vocal_name)

        if gui_screen is not None:

            gui_screen.ids.dataset_progress_status.text = "Data processing complete!"
            gui_screen.ids.dataset_progress_file.text = "Press 'Finish' to return back to menu"
            gui_screen.ids.dataset_finish_button.disabled = False
            gui_screen.ids.dataset_cancel_button.disabled = True
    else:
        spectral_data, aperiodic_data, label_data, cutoff_points, frequency = load_training_data(vocal_name)

    return spectral_data, aperiodic_data, label_data, cutoff_points, frequency


def read_test_data(trained_vocal_name, f_data, compare=False, note_file=False, index_loc="Dataset/Test", de_tune=False,
                   index_name="index.xlsx"):

    spectral_data, aperiodic_data, label_data, cutoff_points, frequency = read_data(index_loc, index_name=index_name)

    data = [spectral_data, aperiodic_data, label_data, cutoff_points, frequency]

    spectral_data, aperiodic_data, label_data, column_list, frequency = process_and_save(data, trained_vocal_name,
                                                                                         save=False)

    label_data = match_input_columns(column_list, label_data)

    if note_file:
        file_type = index_name.split('.')[1]
        note_loc = index_loc + '/notes.' + file_type
        if not os.path.exists(note_loc):
            note_loc = None
    else:
        note_loc = None

    f_label_data, frequency = extract_f_labels(frequency, f_data, label_data, note_file=note_loc, de_tune=de_tune)

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


def read_notes(note_index, audio_length):

    file_name = os.path.basename(note_index)
    file_type = file_name.split('.')[1]

    if file_type == 'xlsx' or file_type == 'xls':
        sound_index = pd.read_excel(note_index, header=None, index_col=False)
    else:
        sound_index = pd.read_csv(note_index, header=None, index_col=False, skip_blank_lines=True)

    note_align = list(sound_index.itertuples())

    step = params.frame_period / 1000
    note_position = 0
    note_array = []
    x = 0

    while x < audio_length:
        if note_align[note_position][1] <= x * step:
            if note_align[note_position][2] > x * step:
                note_array.append(note_align[note_position][3])
                x = x + 1
            elif note_position + 1 < len(note_align):
                note_position = note_position + 1
            else:
                x = x + 1
                note_array.append(note_align[note_position][3])
        else:
            note_array.append("N")
            x = x + 1
    return note_array


def extract_lyrics(artist, title):

    lyric_api = azapi.AZlyrics('google', accuracy=0.5)

    lyric_api.artist = artist
    lyric_api.title = title

    lyrics = lyric_api.getLyrics(save=False)

    return lyrics
