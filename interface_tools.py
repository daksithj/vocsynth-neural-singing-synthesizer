import os
import pandas as pd
from args import parser
from tensorflow.keras.callbacks import Callback

params = parser.parse_args()

SPECTRAL_MODE = 0
APERIODIC_MODE = 1
FREQUENCY_MODE = 2


class GuiCallBack(Callback):

    def __init__(self, gui=None, total_epoch=None, batch_len=None):

        super(GuiCallBack, self).__init__()
        self.gui = gui
        self.total_epoch = total_epoch
        self.batch_len = batch_len

    def on_train_begin(self, logs=None):

        self.gui.ids.train_progress_bar.min = 0
        self.gui.ids.train_progress_bar.max = self.total_epoch

        self.gui.ids.train_epoch_bar.min = 0
        self.gui.ids.train_epoch_bar.max = self.batch_len

        self.gui.ids.train_progress_value.text = f'Epoch: 0/{self.total_epoch}'

    def on_epoch_end(self, epoch, logs=None):

        self.gui.ids.train_progress_value.text = f'Epoch {epoch + 1}/{self.total_epoch} Complete'

        self.gui.ids.train_progress_bar.value = epoch + 1

        if self.gui.kill_signal:
            self.model.stop_training = True

    def on_batch_end(self, batch, logs=None):

        if self.gui.kill_signal:
            self.model.stop_training = True

        loss = logs['loss']
        self.gui.ids.train_progress_status.text = f'Loss: {loss}'
        self.gui.ids.train_epoch_bar.value = batch


def verify_index_file(index_file_location):

    try:
        sound_index = pd.read_excel(index_file_location, header=None, index_col=False)

        directory = os.path.dirname(index_file_location)

        audio_file_count = 0

        for row in sound_index.itertuples():
            lyrics_file_name = directory + '/' + str(row[1]) + '.wav'

            if os.path.isfile(lyrics_file_name):
                audio_file_count += 1
            else:
                return None
            if len(str(row[2])) == 0:
                return None

        if audio_file_count == 0:
            return None

        return audio_file_count
    except:
        return None


def check_dataset_exist(data_name):

    directory = params.training_dir + '/' + data_name
    if os.path.exists(directory):
        return True
    return False
