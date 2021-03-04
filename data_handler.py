from tensorflow.keras.utils import Sequence
import pandas as pd
import numpy as np
from random import randint

from frequency_tools import extract_notes, shift_pitch, get_note_data


class HarmonicDataSet(Sequence):

    def __init__(self,
                 spectral_data_set,
                 aperiodic_data_set,
                 label_data,
                 cutoff_points,
                 data_length=210,
                 noise=1.2,
                 model_type=0):

        self.data_length = data_length
        self.noise = noise
        self.model_type = model_type

        spectral_files = []
        aperiodic_files = []
        label_files = []
        start = 0

        for point in cutoff_points:
            spectral_data = spectral_data_set[start:point]
            aperiodic_data = aperiodic_data_set[start:point]

            spectral_data = np.pad(spectral_data, ((1, 0), (0, 0)))
            aperiodic_data = np.pad(aperiodic_data, ((1, 0), (0, 0)))

            spectral_files.append(np.asarray(spectral_data))
            aperiodic_files.append(np.asarray(aperiodic_data))
            label_files.append(np.asarray(label_data[start:point]))

            start = point

        spectral_array = []
        aperiodic_array = []
        label_array = []

        for sp, ap, label in zip(spectral_files, aperiodic_files, label_files):

            start = 0
            reading = True

            while reading:

                short = data_length - (sp.shape[0] - 1 - start)

                if short > 0:
                    start = start - short
                    reading = False

                label_slice = label[start:start+data_length, :]
                label_array.append(np.asarray([label_slice]))

                spectral_slice = sp[start:start+data_length+1, :]
                spectral_array.append(np.asarray([spectral_slice]))

                aperiodic_slice = ap[start:start + data_length + 1, :]
                aperiodic_array.append(np.asarray([aperiodic_slice]))

                start = start + data_length

        self.spectral_data = np.asarray(spectral_array)
        self.aperiodic_data = np.asarray(aperiodic_array)
        self.label_data = np.asarray(label_array).astype(np.float)

    def __getitem__(self, idx):

        label_input = self.label_data[idx]

        spectral_part = self.spectral_data[idx]
        spectral_input = spectral_part[:, :self.data_length, :]

        if self.model_type == 0:
            model_input = np.random.normal(spectral_input, self.noise, size=spectral_input.shape)
            model_target = np.array(spectral_part[:, -self.data_length:, :])
        else:
            aperiodic_part = self.aperiodic_data[idx]

            model_input = aperiodic_part[:, :self.data_length, :]
            model_input = np.concatenate([model_input, spectral_input], axis=2)
            model_input = np.random.normal(model_input, self.noise, size=model_input.shape)

            model_target = np.array(aperiodic_part[:, -self.data_length:, :])

        return [np.array(model_input), np.array(label_input)], model_target

    def __len__(self):
        return self.label_data.shape[0]

    def get_data_length(self):
        return self.data_length

    def get_label_channels(self):
        return self.label_data.shape[3]

    def get_data_channels(self):
        if self.model_type == 0:
            return self.spectral_data.shape[3]
        else:
            return self.spectral_data.shape[3] + self.aperiodic_data.shape[3]

    def set_noise(self, noise):
        self.noise = noise

    def set_type(self, model_type):
        if model_type == 0 or 1:
            self.model_type = model_type
        else:
            print("Invalid model type. Using model type: " + str(self.model_type))
        return self


class FrequencyDataSet(Sequence):

    def __init__(self,
                 frequency_data_set,
                 label_data,
                 cutoff_points,
                 data_length=105,
                 noise=0.2):

        self.data_length = data_length
        self.noise = noise

        label_data = label_data[:, 256:]
        notes_data_set, self.min_p, self.max_p = extract_notes(frequency_data_set)

        self.mean_p = int(np.round(np.mean(notes_data_set[notes_data_set > -1])))

        frequency_data_set = np.expand_dims(frequency_data_set, axis=1)

        self.min_f = np.min(frequency_data_set)
        self.max_f = np.max(frequency_data_set)

        notes_data_set, note_timings = get_note_data(notes_data_set)
        label_data = np.concatenate([note_timings, label_data], axis=1)

        frequency_files = []
        note_files = []
        label_files = []
        start = 0

        for point in cutoff_points:
            frequency_data = frequency_data_set[start:point]

            frequency_data = np.pad(frequency_data, ((1, 0), (0, 0)))

            frequency_files.append(np.asarray(frequency_data))
            note_files.append(np.asarray(notes_data_set[start:point]))
            label_files.append(np.asarray(label_data[start:point]))

            start = point

        frequency_array = []
        note_array = []
        label_array = []

        for f, nt, label in zip(frequency_files, note_files, label_files):

            start = 0
            reading = True

            while reading:

                short = data_length - (f.shape[0] - 1 - start)

                if short > 0:
                    start = start - short
                    reading = False

                label_slice = label[start:start+data_length, :]
                label_array.append(np.asarray([label_slice]))

                frequency_slice = f[start:start+data_length+1, :]
                frequency_array.append(np.asarray([frequency_slice]))

                note_slice = nt[start:start+data_length, :]
                note_array.append(np.asarray([note_slice]))

                start = start + data_length

        self.frequency_data = np.asarray(frequency_array)
        self.label_data = np.asarray(label_array).astype(np.float)
        self.note_data = np.asarray(note_array)

    def __getitem__(self, idx):

        label_input = self.label_data[idx]
        note_input = self.note_data[idx]

        frequency_part = self.frequency_data[idx]

        frequency_part, note_input = self.shift_data(frequency_part, note_input)

        frequency_input = frequency_part[:, :self.data_length, :]
        model_input = np.random.normal(frequency_input, self.noise, size=frequency_input.shape)

        label_input = np.concatenate([note_input, label_input], axis=2)

        model_target = np.array(frequency_part[:, -self.data_length:, :])

        return [model_input, np.array(label_input)], model_target

    def shift_data(self, frequency_data, note_data, shift=True):

        if shift:
            samp_max = np.max(note_data)

            if samp_max > -1:
                samp_min = np.min(note_data[note_data > -1])
                shift_min = self.min_p - samp_min
                shift_max = self.max_p - samp_max

                p_shift = randint(shift_min, shift_max)

                note_data, frequency_data = shift_pitch(note_data, frequency_data, p_shift)

        note_data = np.append(note_data, [[[-1, -1, -1]]], axis=1)
        for i in range(self.min_p, self.max_p + 1):
            note_data = np.append(note_data, [[[i, i, i]]], axis=1)

        note_data = np.squeeze(note_data, axis=0)
        note_data = pd.DataFrame(note_data)

        note_data = pd.get_dummies(data=note_data, columns=[0], prefix='Note')
        note_data = pd.get_dummies(data=note_data, columns=[1], prefix='Pre_note')
        note_data = pd.get_dummies(data=note_data, columns=[2], prefix='Post_note')

        note_data = np.asarray(note_data)
        note_data = note_data[:-(self.max_p - self.min_p + 2), :]
        note_data = np.expand_dims(note_data, axis=0)

        frequency_data = (frequency_data - self.min_f) / (self.max_f - self.min_f) - 0.5
        return frequency_data, note_data

    def decode_frequency(self, frequency):
        frequency = (frequency + 0.5) * (self.max_f - self.min_f) + self.min_f
        return frequency

    def de_tune(self, notes_data_set, frequency):

        mean_p = int(np.round(np.mean(notes_data_set[notes_data_set > -1])))

        pitch_shift = self.mean_p - mean_p

        notes_data_set, frequency = shift_pitch(notes_data_set, frequency, pitch_shift)

        return notes_data_set, frequency

    def __len__(self):
        return self.label_data.shape[0]

    def get_data_length(self):
        return self.data_length

    def get_label_channels(self):
        return self.label_data.shape[3] + 3 * (self.max_p - self.min_p + 2)

    def get_data_channels(self):
        return self.frequency_data.shape[3]

    def set_noise(self, noise):
        self.noise = noise

    def get_params(self):
        return self.min_f, self.max_f, self.min_p, self.max_p
