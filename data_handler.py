from tensorflow.keras.utils import Sequence
import numpy as np


class HarmonicDataSet(Sequence):

    def __init__(self,
                 spectral_data_set,
                 aperiodic_data_set,
                 label_data,
                 cutoff_points,
                 data_length=210,
                 noise=1.0,
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
