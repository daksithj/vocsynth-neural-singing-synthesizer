from model import SingingModel
from synthesize import construct_audio, decode_envelopes
from read_data import read_test_data, read_training_data
from args import parser

params = parser.parse_args()

if __name__ == '__main__':

    SPECTRAL_MODE = 0
    APERIODIC_MODE = 1

    model_name = params.model_name
    output_name = params.output_name

    load_data = params.load_data

    sp_train = params.sp_train
    sp_cont = params.sp_cont

    ap_train = params.ap_train
    ap_cont = params.ap_cont

    spectral_data, aperiodic_data, label_data, cutoff_points = read_training_data(model_name, load=load_data)

    singing_model = SingingModel(spectral_data, aperiodic_data, label_data, cutoff_points, model_name)

    if sp_train:
        singing_model.train_model(SPECTRAL_MODE, sp_cont)

    if ap_train:
        singing_model.train_model(APERIODIC_MODE, ap_cont)

    label_data, frequency = read_test_data(model_name)

    spectral_output = singing_model.inference(label_data,  SPECTRAL_MODE)

    aperiodic_output = singing_model.inference(label_data, APERIODIC_MODE, spectral_output)

    spectral_output, aperiodic_output = decode_envelopes(spectral_output, aperiodic_output, params.sample_rate,
                                                         model_name)

    construct_audio(spectral_output, aperiodic_output, frequency, output_name)
