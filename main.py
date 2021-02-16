from model import SingingModel
import numpy as np
from synthesize import construct_audio, decode_envelopes
from read_data import read_test_data, read_training_data, add_frequency_data
from data_handler import FrequencyDataSet
from args import parser

params = parser.parse_args()

if __name__ == '__main__':

    SPECTRAL_MODE = 0
    APERIODIC_MODE = 1
    FREQUENCY_MODE = 2

    model_name = params.model_name
    output_name = params.output_name

    load_data = params.load_data

    sp_train = params.sp_train
    sp_cont = params.sp_cont
    sp_use = params.sp_use

    ap_train = params.ap_train
    ap_cont = params.ap_cont
    ap_use = params.ap_use

    f_train = params.f_train
    f_cont = params.f_cont
    f_use = params.f_use

    spectral_data, aperiodic_data, label_data, cutoff_points, frequency = read_training_data(model_name, load=load_data)

    f_data = FrequencyDataSet(frequency, label_data, cutoff_points)

    singing_model = SingingModel(spectral_data, aperiodic_data, f_data, label_data, cutoff_points, model_name)

    if f_train:
        singing_model.train_model(FREQUENCY_MODE, f_cont)

    if sp_train:
        singing_model.train_model(SPECTRAL_MODE, sp_cont)

    if ap_train:
        singing_model.train_model(APERIODIC_MODE, ap_cont)

    f_data = FrequencyDataSet(frequency, label_data, cutoff_points)

    label_data, f_label_data, frequency = read_test_data(model_name, f_data)

    if f_use:
        frequency = singing_model.inference(f_label_data, FREQUENCY_MODE)
        frequency = np.squeeze(frequency)

        frequency = f_data.decode_frequency(frequency)

        label_data = add_frequency_data(label_data, frequency)

    spectral_output = singing_model.inference(label_data, SPECTRAL_MODE)

    aperiodic_output = singing_model.inference(label_data, APERIODIC_MODE, spectral_output)

    spectral_output, aperiodic_output = decode_envelopes(spectral_output, aperiodic_output, params.sample_rate,
                                                         model_name)

    construct_audio(spectral_output, aperiodic_output, frequency, output_name)
