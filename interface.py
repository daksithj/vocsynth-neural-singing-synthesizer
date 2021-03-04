from kivy.app import App
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.core.audio import SoundLoader
from kivy.uix.screenmanager import ScreenManager, Screen, RiseInTransition, ScreenManagerException
from kivy.uix.popup import Popup
from interface_tools import verify_index_file, check_dataset_exist
from read_data import read_test_data, read_training_data, add_frequency_data
from frequency_tools import smooth_out
from data_handler import FrequencyDataSet
from synthesize import construct_audio, decode_envelopes
from model import SingingModel
from args import parser, h_parser, a_parser, f_parser
from threading import Thread
import win32api
import numpy as np
import os


Window.size = (1024, 768)

SPECTRAL_MODE = 0
APERIODIC_MODE = 1
FREQUENCY_MODE = 2


class StartWindow(Screen):

    def __init__(self, **kwargs):

        super(StartWindow, self).__init__(**kwargs)
        self.train_window = TrainWindow()
        self.generate_window = GenerateWindow()

    def on_train(self):

        try:
            self.manager.get_screen("train_window")
        except ScreenManagerException:
            self.manager.add_widget(self.train_window)

        self.manager.current = "train_window"

    def on_generate(self):

        try:
            self.manager.get_screen("generate_window")
        except ScreenManagerException:
            self.manager.add_widget(self.generate_window)

        self.manager.current = "generate_window"


class TrainWindow(Screen):

    def __init__(self, **kwargs):

        super(TrainWindow, self).__init__(**kwargs)
        self.dataset_window = DatasetWindow()
        self.train_window = TrainModelWindow()

    def open_train(self):

        try:
            self.manager.get_screen("train_model_window")
        except ScreenManagerException:
            self.manager.add_widget(self.train_window)

        self.manager.current = "train_model_window"

    def open_dataset(self):

        try:
            self.manager.get_screen('dataset_window')
        except ScreenManagerException:
            self.manager.add_widget(self.dataset_window)

        self.manager.current = 'dataset_window'


class DatasetWindow(Screen):

    def __init__(self, **kwargs):

        super(DatasetWindow, self).__init__(**kwargs)
        self.verify_window = DatasetVerifyWindow()

    def on_pre_enter(self, *args):

        self.ids.index_file_chooser.path = '.'
        self.ids.index_file_chooser_drive.text = 'Choose drive'
        self.ids.dataset_selected_file.text = ''

    def update_file_path_dir(self):

        drive = self.ids.index_file_chooser_drive.text
        if drive == 'Choose drive':
            self.ids.index_file_chooser.path = '.'
        else:
            self.ids.index_file_chooser.path = drive

    def update_drives(self):

        drives = win32api.GetLogicalDriveStrings()
        drives = drives.split('\000')[:-1]
        self.ids.index_file_chooser_drive.values = drives

    def on_select_file(self, file_name):

        try:
            self.ids.dataset_selected_file.text = file_name[0]
        except IndexError:
            pass

    def on_submit_file(self):

        index_file = self.ids.dataset_selected_file.text

        if index_file == "":
            self.ids.dataset_wrong_file.text = "Please select a valid index file!"
        else:
            count = verify_index_file(index_file)

            if count is None:
                self.ids.dataset_wrong_file.text = "Please select a valid index file!"
            else:

                try:
                    self.manager.get_screen('dataset_verify_window')
                except ScreenManagerException:
                    self.manager.add_widget(self.verify_window)

                self.verify_window.ids.dataset_verify_location.text = index_file
                self.verify_window.ids.dataset_verify_count.text = str(count)
                self.manager.current = 'dataset_verify_window'

    def on_back(self):

        self.manager.current = "train_window"


class DatasetVerifyWindow(Screen):

    def __init__(self, **kwargs):

        super(DatasetVerifyWindow, self).__init__(**kwargs)
        self.verify_popup = VerifyPopUp()
        self.pending_window = DatasetPendingWindow()

    def on_pre_enter(self, *args):

        self.ids.dataset_verify_name.text = ""
        self.ids.dataset_wrong_file.text = ""

    def start_process(self):

        data_name = self.ids.dataset_verify_name.text

        if data_name == "":

            self.ids.dataset_wrong_file.text = "Please select a valid dataset name!"

        else:
            read_args = {'vocal_name': data_name,
                         'data_dir': self.ids.dataset_verify_location.text}

            if check_dataset_exist(data_name):
                self.verify_popup.set_pop_up(read_args, self)
                self.verify_popup.open()

            else:
                self.start_pending(read_args)

    def start_pending(self, read_args):

        try:
            self.manager.get_screen('dataset_pending_window')
        except ScreenManagerException:
            self.manager.add_widget(self.pending_window)

        self.manager.current = 'dataset_pending_window'
        self.pending_window.start_generating(read_args)

    def on_back(self):

        self.manager.current = 'dataset_window'


class VerifyPopUp(Popup):

    def __init__(self, **kwargs):

        super(VerifyPopUp, self).__init__(**kwargs)
        self.read_args = None
        self.verify_window = None

    def set_pop_up(self, read_args, verify_window):

        self.read_args = read_args
        self.verify_window = verify_window

    def overwrite(self):

        self.dismiss()
        self.verify_window.start_pending(self.read_args)


class DatasetPendingWindow(Screen):

    def __init__(self, **kwargs):

        super(DatasetPendingWindow, self).__init__(**kwargs)
        self.kill_signal = False

    def on_pre_enter(self, *args):

        self.ids.dataset_progress_status.text = "Please wait until the data set is processed"
        self.ids.dataset_progress_value.text = "0% Progress"
        self.ids.dataset_progress_bar.value = 0
        self.ids.dataset_progress_file.text = "Processing"
        self.ids.dataset_finish_button.disabled = True

    def start_generating(self, read_args):

        data_dir = read_args['data_dir']
        base_name = os.path.basename(data_dir)
        data_dir = os.path.dirname(data_dir)
        data_dir = data_dir.replace('\\', '/')

        Thread(target=read_training_data,
               args=(data_dir, read_args['vocal_name'], base_name, self, False),
               daemon=True).start()

    def cancel_process(self):

        self.kill_signal = True
        self.ids.dataset_progress_status.text = "Please wait until process is cancelled!"


class TrainModelWindow(Screen):

    def __init__(self, **kwargs):

        super(TrainModelWindow, self).__init__(**kwargs)

        self.file_list = ['aperiodic_data.npy', 'column_list.pkl', 'cutoff_points.npy', 'frequency.npy',
                          'label_data.npy', 'min_max.npy', 'spectral_data.npy']

        self.train_params = {'train_s': False,
                             'train_a': False,
                             'train_f': False,
                             'm_param': parser.parse_args(),
                             'h_param': h_parser.parse_args(),
                             'a_param': a_parser.parse_args(),
                             'f_param': f_parser.parse_args(),
                             'data_set': None,
                             'model_name': None,
                             'h_cont': False,
                             'a_cont': False,
                             'f_cont': False
                             }

        self.train_pending_window = TrainingPendingWindow()

    def on_pre_enter(self, *args):

        self.ids.dataset_chooser.text = 'Choose a dataset'
        self.ids.train_spectral_switch.active = False
        self.ids.harmonic_param_button.disabled = True
        self.ids.train_aperiodic_switch.active = False
        self.ids.aperiodic_param_button.disabled = True
        self.ids.train_frequency_switch.active = False
        self.ids.frequency_param_button.disabled = True

        self.ids.start_train_button.disabled = True

        self.train_params['h_param'] = h_parser.parse_args()
        self.train_params['a_param'] = a_parser.parse_args()
        self.train_params['f_param'] = f_parser.parse_args()

    def validate_data(self):

        invalid_choices = ['', 'Choose a dataset']

        if self.ids.dataset_chooser.text in invalid_choices:

            self.ids.start_train_button.disabled = True
            self.ids.cont_train_button.disabled = True
            return

        if self.train_params['train_s'] or self.train_params['train_a'] or self.train_params['train_f']:

            self.ids.start_train_button.disabled = False

            if self.validate_model_exist():
                self.ids.cont_train_button.disabled = False

            else:
                self.ids.cont_train_button.disabled = True

        else:
            self.ids.start_train_button.disabled = True
            self.ids.cont_train_button.disabled = True

    def validate_model_exist(self):

        directory = self.train_params['m_param'].model_dir + '/' + self.ids.dataset_chooser.text

        if self.train_params['train_s']:

            file_name = directory + '/' + 'harmonic_model.h5'

            if os.path.exists(file_name):
                return True

        if self.train_params['train_a']:

            file_name = directory + '/' + 'aperiodic_model.h5'

            if os.path.exists(file_name):
                return True

        if self.train_params['train_f']:

            file_name = directory + '/' + 'frequency_model.h5'

            if os.path.exists(file_name):
                return True

        return False

    def list_data_sets(self):

        data_dir = parser.parse_args().training_dir

        dir_list = []

        for dir_in_list in os.listdir(data_dir):

            directory = data_dir + '/' + dir_in_list

            if os.path.isdir(directory):

                dir_file_list = os.listdir(directory)

                if set(dir_file_list) == set(self.file_list):

                    dir_list.append(dir_in_list)

        self.ids.dataset_chooser.values = dir_list

    def toggle_button(self, button_type, _, value):

        if button_type == 'harmonic':

            button = self.ids.harmonic_param_button
            train_param = 'train_s'

        elif button_type == 'aperiodic':

            button = self.ids.aperiodic_param_button
            train_param = 'train_a'

        else:

            button = self.ids.frequency_param_button
            train_param = 'train_f'

        if value:

            button.disabled = False
            self.train_params[train_param] = True

        else:

            button.disabled = True
            self.train_params[train_param] = False

        self.validate_data()

    def param_adjust(self, param_type):

        settings = ModelParamPopUp(param_type, self.train_params)

        settings.open()

    def on_submit(self):

        self.train_params['data_set'] = self.ids.dataset_chooser.text
        self.train_params['model_name'] = self.ids.dataset_chooser.text

        try:
            self.manager.get_screen('train_pending_window')
        except ScreenManagerException:
            self.manager.add_widget(self.train_pending_window)

        self.manager.current = 'train_pending_window'
        self.train_pending_window.start_train(self.train_params)

    def on_cont(self):

        directory = self.train_params['m_param'].model_dir + '/' + self.ids.dataset_chooser.text

        if self.train_params['train_s']:

            file_name = directory + '/' + 'harmonic_model.h5'

            if os.path.exists(file_name):

                self.train_params['h_cont'] = True

        if self.train_params['train_a']:

            file_name = directory + '/' + 'aperiodic_model.h5'

            if os.path.exists(file_name):

                self.train_params['a_cont'] = True

        if self.train_params['train_f']:

            file_name = directory + '/' + 'frequency_model.h5'

            if os.path.exists(file_name):

                self.train_params['f_cont'] = True

        self.on_submit()

    def go_back(self):

        self.manager.current = 'train_window'


class ModelParamPopUp(Popup):

    def __init__(self,  model_type=None, model_params=None, **kwargs):

        super().__init__(**kwargs)

        if model_type == 'harmonic':

            self.title = 'Spectral model parameters'
            self.params = model_params['h_param']

        elif model_type == 'aperiodic':

            self.title = 'Aperiodic model parameters'
            self.params = model_params['a_param']

        else:

            self.title = 'Frequency model parameters'
            self.params = model_params['f_param']

        self.ids.param_levels.text = str(self.params.levels)
        self.ids.param_blocks.text = str(self.params.blocks)
        self.ids.param_epochs.text = str(self.params.epochs)
        self.ids.param_l2.text = str(self.params.l2_decay)
        self.ids.param_lr.text = str(self.params.learn_rate)
        self.ids.param_temp.text = str(self.params.temp)

    def check_value(self, value, data_type, min_val, max_val):

        try:
            if data_type == 'int':
                data = int(value)

            else:
                data = float(value)

            if min_val <= data < max_val:
                self.ids.param_confirm_button.disabled = False

            else:
                self.ids.param_confirm_button.disabled = True

        except ValueError:
            self.ids.param_confirm_button.disabled = True

    def on_submit(self):

        self.params.levels = int(self.ids.param_levels.text)
        self.params.blocks = int(self.ids.param_blocks.text)
        self.params.epochs = int(self.ids.param_epochs.text)
        self.params.l2_decay = float(self.ids.param_l2.text)
        self.params.learn_rate = float(self.ids.param_lr.text)
        self.params.temp = float(self.ids.param_temp.text)

        self.dismiss()


class TrainingPendingWindow(Screen):

    def __init__(self, **kwargs):

        super(TrainingPendingWindow, self).__init__(**kwargs)
        self.params = None
        self.kill_signal = False

    def start_train(self, params):

        self.kill_signal = False
        self.params = params
        Thread(target=self.start_generating, daemon=True).start()

    def start_generating(self):

        data_dir = self.params['m_param'].data_dir + '/' + self.params['data_set']
        model_name = self.params['model_name']
        data_set = self.params['data_set']

        parameters = (self.params['m_param'], self.params['h_param'], self.params['a_param'], self.params['f_param'])

        spectral_data, aperiodic_data, label_data, cutoff_points, frequency = read_training_data(data_dir,
                                                                                                 data_set,
                                                                                                 load=True)

        f_data = FrequencyDataSet(frequency, label_data, cutoff_points)
        singing_model = SingingModel(spectral_data, aperiodic_data, f_data, label_data, cutoff_points, model_name,
                                     parameters=parameters, train_gui=self)

        if self.params['train_f']:
            self.ids.train_progress_bar.value = 0
            self.ids.train_epoch_bar.value = 0
            self.ids.train_progress_state.text = 'Training Frequency Model'
            singing_model.train_model(FREQUENCY_MODE, load=self.params['f_cont'])

        if self.params['train_s']:
            self.kill_signal = False
            self.ids.train_progress_bar.value = 0
            self.ids.train_epoch_bar.value = 0
            self.ids.train_progress_state.text = 'Training Spectral Model'
            singing_model.train_model(SPECTRAL_MODE, load=self.params['h_cont'])

        if self.params['train_a']:
            self.kill_signal = False
            self.ids.train_progress_bar.value = 0
            self.ids.train_epoch_bar.value = 0
            self.ids.train_progress_state.text = 'Training Aperiodic Model'
            singing_model.train_model(APERIODIC_MODE, load=self.params['a_cont'])

        if self.kill_signal:
            self.ids.train_progress_state.text = "Training Cancelled!"

        else:
            self.ids.train_progress_state.text = "Training Completed!"

        self.ids.train_progress_state.text = "Press Finish to return to menu"

        self.ids.train_finish_button.disabled = False
        self.ids.train_cancel_button.disabled = True

    def cancel_process(self):

        self.kill_signal = True
        self.ids.train_progress_state.text = "Please wait until training process is cancelled!"

    def on_finish(self):

        self.manager.current = "train_window"
        self.ids.train_progress_bar.value = 0
        self.ids.train_epoch_bar.value = 0

        self.ids.train_finish_button.disabled = True
        self.ids.train_cancel_button.disabled = False
        self.ids.train_progress_state.text = "Please wait until the model is trained"
        self.ids.train_progress_status.text = "Starting training"
        self.ids.train_progress_value.text = "Epochs: "


class GenerateWindow(Screen):

    def __init__(self, **kwargs):

        super(GenerateWindow, self).__init__(**kwargs)
        self.verify_window = GenerateVerifyWindow()

    def on_pre_enter(self, *args):

        self.ids.index_file_chooser_drive.text = 'Choose drive'
        self.ids.index_file_chooser.path = '.'
        self.ids.gen_selected_file.text = ""
        self.ids.gen_wrong_file = ""

    def update_drives(self):

        drives = win32api.GetLogicalDriveStrings()
        drives = drives.split('\000')[:-1]
        self.ids.index_file_chooser_drive.values = drives

    def on_select_file(self, file_name):

        try:
            self.ids.gen_selected_file.text = file_name[0]
        except IndexError:
            pass

    def on_back(self):

        self.manager.current = "start_window"

    def on_submit_file(self):

        index_file = self.ids.gen_selected_file.text

        if index_file == "":
            self.ids.gen_wrong_file.text = "Please select a valid index file!"

        else:
            count = verify_index_file(index_file)

            if count is None:
                self.ids.gen_wrong_file.text = "Please select a valid index file!"

            else:
                index_directory = os.path.dirname(index_file)
                note_file = index_directory + '/' + 'notes.xlsx'

                if os.path.exists(note_file):
                    self.verify_window.custom_notes = True

                self.verify_window.ids.gen_verify_location.text = index_file
                self.verify_window.ids.gen_verify_count.text = str(count)

                try:
                    self.manager.get_screen('generate_verify_window')
                except ScreenManagerException:
                    self.manager.add_widget(self.verify_window)

                self.manager.current = 'generate_verify_window'


class GenerateVerifyWindow(Screen):

    def __init__(self, **kwargs):

        super(GenerateVerifyWindow, self).__init__(**kwargs)
        self.use_f = False
        self.m_param = parser.parse_args()
        self.custom_notes = False
        self.use_custom_notes = False
        self.use_change_key = True
        self.smoothing = parser.parse_args().f_smooth
        self.reverb = parser.parse_args().reverb
        self.gen_pending_window = GeneratePendingWindow()

    def on_pre_enter(self, *args):

        self.ids.model_chooser.text = 'Choose a model'
        self.ids.use_frequency_switch.disabled = True
        self.ids.use_frequency_switch.active = True

        self.set_frequency_options(True)

        self.ids.use_reverb.active = True
        self.ids.reverb_amount.text = str(parser.parse_args().reverb)

        self.ids.gen_output_name.text = ""
        self.ids.start_gen_button.disabled = True

    def list_data_sets(self):

        data_dir = parser.parse_args().model_dir

        dir_list = []

        for dir_in_list in os.listdir(data_dir):

            directory = data_dir + '/' + dir_in_list

            if os.path.isdir(directory):
                dir_file_list = os.listdir(directory)

                if 'harmonic_model.h5' in dir_file_list and 'aperiodic_model.h5' in dir_file_list:
                    dir_list.append(dir_in_list)

        self.ids.model_chooser.values = dir_list

    def set_frequency_options(self, disabled):

        if self.custom_notes:
            self.ids.use_custom_notes.disabled = disabled

        if disabled:
            self.ids.use_custom_notes.disabled = disabled

        self.ids.use_change_key.disabled = disabled
        self.ids.use_smoothing.disabled = disabled
        self.ids.smoothing_amount.disabled = disabled

        self.ids.use_custom_notes.active = False
        self.use_custom_notes = False

        self.ids.use_change_key.active = True
        self.use_change_key = True

        self.ids.use_smoothing.active = True
        self.ids.smoothing_amount.text = str(parser.parse_args().f_smooth)
        self.smoothing = parser.parse_args().f_smooth

    def on_select_model(self):

        model_name = self.ids.model_chooser.text
        f_model = parser.parse_args().model_dir + '/' + model_name + '/frequency_model.h5'

        if os.path.exists(f_model):
            self.ids.use_frequency_switch.disabled = False
            self.ids.use_frequency_switch.active = True
            self.use_f = True

            self.set_frequency_options(False)

        else:
            self.ids.use_frequency_switch.disabled = True
            self.ids.use_frequency_switch.active = False
            self.use_f = False

            self.set_frequency_options(True)

        self.validate()

    def reverb_toggle(self, _, value):

        if value:
            self.ids.reverb_amount.disabled = False
            self.ids.reverb_amount.text = str(parser.parse_args().reverb)
            self.reverb = parser.parse_args().reverb

        else:
            self.ids.reverb_amount.disabled = True
            self.ids.reverb_amount.text = '0'
            self.reverb = 0

    def frequency_toggle(self, _, value):

        if value:
            self.use_f = True
            self.set_frequency_options(False)

        else:
            self.use_f = False
            self.set_frequency_options(True)

    def custom_note_toggle(self, _, value):

        if value:
            if self.custom_notes:
                self.use_custom_notes = True

        else:
            self.use_custom_notes = False

    def change_key_toggle(self, _, value):

        if value:
            self.use_change_key = True

        else:
            self.use_change_key = False

    def smoothing_toggle(self, _, value):

        if value:
            self.ids.smoothing_amount.disabled = False
            self.ids.smoothing_amount.text = str(parser.parse_args().f_smooth)
            self.smoothing = parser.parse_args().f_smooth

        else:
            self.ids.smoothing_amount.disabled = True
            self.ids.smoothing_amount.text = '0'
            self.smoothing = 0

    def validate(self):

        invalid_choices = ['', 'Choose a model']

        if self.ids.model_chooser.text in invalid_choices:
            self.ids.start_gen_button.disabled = True
            return

        if self.ids.gen_output_name.text == '':
            self.ids.start_gen_button.disabled = True
            return

        if self.reverb != 0:
            try:
                data = int(self.ids.reverb_amount.text)

                if 0 <= data < 100:
                    self.ids.start_gen_button.disabled = False
                    self.reverb = data
                else:
                    self.ids.start_gen_button.disabled = True
            except ValueError:
                self.ids.start_gen_button.disabled = True

        if self.smoothing != 0:
            try:
                data = int(self.ids.smoothing_amount.text)

                if 0 <= data < 1000:
                    self.ids.start_gen_button.disabled = False
                    self.smoothing = data
                else:
                    self.ids.start_gen_button.disabled = True
            except ValueError:
                self.ids.start_gen_button.disabled = True

        else:
            self.ids.start_gen_button.disabled = False

    def on_submit(self):

        params = {
            'use_f': self.use_f,
            'reverb': self.reverb,
            'custom_notes': self.use_custom_notes,
            'change_key': self.use_change_key,
            'smoothing': self.smoothing,
            'model_name': self.ids.model_chooser.text,
            'output_name': self.ids.gen_output_name.text,
            'data_location': self.ids.gen_verify_location.text
        }

        try:
            self.manager.get_screen('generate_pending_window')
        except ScreenManagerException:
            self.manager.add_widget(self.gen_pending_window)

        self.manager.current = 'generate_pending_window'
        self.gen_pending_window.start_gen(params)

    def on_back(self):
        self.manager.current = 'generate_window'


class GeneratePendingWindow(Screen):

    def __init__(self, **kwargs):

        super(GeneratePendingWindow, self).__init__(**kwargs)
        self.params = None
        self.kill_signal = False
        self.sound = None
        self.sound_updater = None
        self.sound_state = False
        self.sound_pos = 0

    def on_pre_enter(self, *args):

        self.ids.gen_progress_state.text = "Starting vocal generation"
        self.ids.f_progress_value.text = "Please wait for Frequency generation"
        self.ids.f_progress_bar.value = 0
        self.ids.s_progress_value.text = "Please wait for Spectral Envelope generation"
        self.ids.s_progress_bar.value = 0
        self.ids.a_progress_value.text = "Please wait for Aperiodic Envelope generation"
        self.ids.a_progress_bar.value = 0
        self.ids.audio_ready.text = ''
        self.ids.audio_slider.value = 0
        self.ids.audio_play_button.disabled = True
        self.ids.audio_stop_button.disabled = True
        self.ids.gen_finish_button.disabled = True
        self.ids.gen_cancel_button.disabled = False
        self.kill_signal = False
        self.sound = None
        self.sound_updater = None
        self.sound_state = False
        self.sound_pos = 0

    def start_gen(self, params):

        self.kill_signal = False
        self.params = params
        Thread(target=self.start_generating, daemon=True).start()

    def start_generating(self):

        m_param = parser.parse_args()
        h_param = h_parser.parse_args()
        a_param = a_parser.parse_args()
        f_param = f_parser.parse_args()

        model_name = self.params['model_name']
        data_dir = m_param.data_dir + '/' + model_name

        parameters = (m_param, h_param, a_param, f_param)

        spectral_data, aperiodic_data, label_data, cutoff_points, frequency = read_training_data(data_dir,
                                                                                                 model_name,
                                                                                                 load=True)
        f_data = FrequencyDataSet(frequency, label_data, cutoff_points)
        singing_model = SingingModel(spectral_data, aperiodic_data, f_data, label_data, cutoff_points, model_name,
                                     parameters=parameters, gen_gui=self)

        test_dir = os.path.dirname(self.params['data_location'])

        self.ids.gen_progress_state.text = 'Reading and processing input'
        label_data, f_label_data, original_frequency = read_test_data(model_name,
                                                                      f_data,
                                                                      note_file=self.params['custom_notes'],
                                                                      index_loc=test_dir,
                                                                      de_tune=self.params['change_key'])

        if not self.kill_signal:

            if self.params['use_f']:
                self.ids.gen_progress_state.text = 'Generating Frequency output'
                frequency = singing_model.inference(f_label_data, FREQUENCY_MODE)

                if not self.kill_signal:
                    frequency = np.squeeze(frequency)
                    frequency = f_data.decode_frequency(frequency)
                    if self.params['smoothing'] > 0:
                        frequency = smooth_out(original_frequency, frequency, self.params['smoothing'])
                    label_data = add_frequency_data(label_data, frequency)

            else:
                self.ids.f_progress_value.text = 'Skipped Frequency generation'

        if not self.kill_signal:
            self.ids.gen_progress_state.text = 'Generating Spectral Envelope'
            spectral_output = singing_model.inference(label_data, SPECTRAL_MODE)

        else:
            self.ids.f_progress_value.text = 'Cancelled generating Frequency output'
            self.ids.s_progress_value.text = 'Cancelled generating Spectral Envelope'

            spectral_output = None

        if not self.kill_signal:
            self.ids.gen_progress_state.text = 'Generating Aperiodic Envelope'
            aperiodic_output = singing_model.inference(label_data, APERIODIC_MODE, spectral_output)

        else:
            self.ids.a_progress_value.text = 'Cancelled generating Aperiodic Envelope'
            aperiodic_output = None

        if not self.kill_signal:
            self.ids.gen_progress_state.text = 'Processing and constructing audio'
            spectral_output, aperiodic_output = decode_envelopes(spectral_output, aperiodic_output, m_param.sample_rate,
                                                                 model_name)
            construct_audio(spectral_output, aperiodic_output, frequency, self.params['output_name'],
                            reverb=self.params['reverb'])
            self.ids.gen_progress_state.text = 'Vocal generation complete!'
            self.audio_ready()

        else:
            self.ids.gen_progress_state.text = 'Cancelled vocal generation'

        self.ids.gen_finish_button.disabled = False
        self.ids.gen_cancel_button.disabled = True

    def audio_ready(self):

        audio_loc = parser.parse_args().output_dir + '/' + self.params['output_name'] + '.wav'
        sound = SoundLoader.load(audio_loc)
        self.sound = sound
        self.sound_updater = None

        if sound:
            self.ids.audio_ready.text = 'Vocal output is ready to be played'
            self.ids.audio_play_button.disabled = False
            self.ids.audio_slider.min = 0
            self.ids.audio_slider.max = sound.length
            sound.bind(on_stop=self.stop_audio)

    def play_audio(self):

        self.sound.play()

        if self.sound_updater is None:
            self.sound_updater = Clock.schedule_interval(self.update_seeker, 0.5)

        self.ids.audio_ready.text = 'Playing generated output'
        self.sound_state = True
        self.ids.audio_play_button.disabled = True
        self.ids.audio_stop_button.disabled = False
        self.sound_pos = 0

    def stop_audio(self, _):

        self.ids.audio_ready.text = 'Vocal output is ready to be played'

        self.sound.stop()
        self.sound_state = False
        self.ids.audio_play_button.disabled = False
        self.ids.audio_stop_button.disabled = True

        self.ids.audio_slider.value = 0
        self.sound_pos = 0

    def update_seeker(self, timer):

        self.sound_pos += timer
        self.ids.audio_slider.value = self.sound_pos

        if not self.sound_state and self.sound_updater is not None:
            self.sound_updater.cancel()
            self.sound_updater = None
            self.ids.audio_slider.value = 0
            self.sound_pos = 0

    def cancel_process(self):

        self.kill_signal = True
        self.ids.gen_progress_state.text = "Please wait until generation is cancelled!"

    def on_finish(self):

        self.manager.current = 'start_window'
        try:
            self.stop_audio(None)
        except AttributeError:
            pass


class WindowManager(ScreenManager):

    def __init__(self, **kwargs):

        super(WindowManager, self).__init__(**kwargs)

        self.transition = RiseInTransition()
        self.add_widget(StartWindow())
        self.current = 'start_window'


class SynthesizeApp(App):

    def build(self):
        self.title = 'VocSynth'
        self.icon = 'Resources/app_logo.png'
        Builder.load_file('interface.kv')
        wm = WindowManager()
        return wm


if __name__ == '__main__':
    SynthesizeApp().run()
