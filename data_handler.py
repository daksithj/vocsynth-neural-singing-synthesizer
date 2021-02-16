from tensorflow.keras.utils import Sequence
import pandas as pd
import numpy as np
from math import log2
from random import randint


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

        self.glob_min = 24
        self.glob_max = 96

        label_data = label_data[:, 256:]
        notes_data_set, self.min_p, self.max_p = self.extract_notes(frequency_data_set)

        frequency_data_set = np.expand_dims(frequency_data_set, axis=1)

        self.min_f = np.min(frequency_data_set)
        self.max_f = np.max(frequency_data_set)

        notes_data_set, note_timings = self.get_note_data(notes_data_set)
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

    def shift_data(self, frequency_data, note_data):
        samp_max = np.max(note_data)
        if samp_max > -1:
            samp_min = np.min(note_data[note_data > -1])
            shift_min = self.min_p - samp_min
            shift_max = self.max_p - samp_max

            p_shift = randint(shift_min, shift_max)
            note_data = note_data + p_shift
            note_data[note_data == (p_shift-1)] = -1

            f_shift = p_shift / 12
            frequency_data = frequency_data * pow(2, f_shift)

        note_data = np.append(note_data, [[[-1, -1, -1]]], axis =1)
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

    @staticmethod
    def extract_notes(f_data):
        A4 = 440
        C0 = A4 * pow(2, -4.75)
        min = 1000
        max = -1
        notes = []
        for freq in f_data:
            if freq > 0:
                h = round(12 * log2(freq / C0))
                notes.append(h)
                if h > max:
                    max = h
                if h < min:
                    min = h
            else:
                notes.append(-1)
        notes = np.asarray(notes)
        return notes, min, max

    @staticmethod
    def get_note_data(notes):
        audio_length = notes.shape[0]

        x = 0
        timing_array = []

        while x < audio_length:
            note = notes[x]
            counter = 0
            while x + counter < audio_length and notes[x + counter] == note:
                counter = counter + 1
            numerator = 1
            for y in range(counter):

                timing = numerator / counter
                if timing <= 0.333:
                    timing_array.append(0)
                elif timing <= 0.666:
                    timing_array.append(1)
                else:
                    timing_array.append(2)
                numerator = numerator + 1
            x = x + counter

        pre_note_array = []
        post_note_array = []
        for y in range(audio_length):
            if y > 0:
                pre_note_array.append(notes[y - 1])
            else:
                pre_note_array.append(-1)
            if y + 1 < audio_length:
                post_note_array.append(notes[y + 1])
            else:
                post_note_array.append(-1)

        notes = np.asarray(notes)
        notes = np.expand_dims(notes, axis =1)
        pre_note_array = np.asarray(pre_note_array)
        pre_note_array = np.expand_dims(pre_note_array, axis=1)
        post_note_array = np.asarray(post_note_array)
        post_note_array = np.expand_dims(post_note_array, axis=1)

        note_data = np.concatenate([notes, pre_note_array, post_note_array], axis=1)

        note_timing_data = pd.DataFrame(timing_array, columns=['Note_timings'])
        note_timing_data = pd.get_dummies(data=note_timing_data, prefix='Note_timings')
        note_timings = np.asarray(note_timing_data)

        return note_data, note_timings

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
