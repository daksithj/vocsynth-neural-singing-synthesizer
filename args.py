import argparse

parser = argparse.ArgumentParser()

# *********** Main Arguments ****************** #
# Name of the vocal model
parser.add_argument("--model_name", type=str, default="Freddy")

# Name of the vocal model
parser.add_argument("--output_name", type=str, default='Freddy-Output')

# Load the data or read the data
parser.add_argument("--load_data", type=bool, default=True)

# Train the Spectral model
parser.add_argument("--sp_train", type=bool, default=True)

# Continue training existing Spectral model (only applies if above is true)
parser.add_argument("--sp_cont", type=bool, default=True)

# Use harmonic model during inference
parser.add_argument("--sp_use", type=bool, default=True)

# Train the Aperiodic model
parser.add_argument("--ap_train", type=bool, default=False)

# Continue training existing Aperiodic model (only applies if above is true)
parser.add_argument("--ap_cont", type=bool, default=False)

# Use aperiodic model during inference
parser.add_argument("--ap_use", type=bool, default=True)

# Train the frequency model
parser.add_argument("--f_train", type=bool, default=False)

# Continue training existing Aperiodic model (only applies if above is true)
parser.add_argument("--f_cont", type=bool, default=True)

# Use pitch model during inference
parser.add_argument("--f_use", type=bool, default=True)


# ************************************************************* #
#       Below are the model related arguments                   #
# ************************************************************* #
# ************ Extracting data arguments **************** #

# Sample rate to convert to after extracting
parser.add_argument("--sample_rate", type=int, default=32000)

# Minimum frequency floor for extracting frequency
parser.add_argument("--min_freq", type=float, default=50.0)
# Maximum frequency ceiling for extracting frequency
parser.add_argument("--max_freq", type=float, default=1100.0)
# Frame period for extracting data
parser.add_argument("--frame_period", type=float, default=5.0)

# ************ Pre-processing arguments **************** #

# **** Frequency to Mel frequency arguments ***** #

# Constant such that Constant in Hz = Constant in Mel
parser.add_argument("--mel_c", type=int, default=1127)
# Corner frequency
parser.add_argument("--corner_freq", type=int, default=700)
# Bin size for mel frequencies
parser.add_argument("--f_bin", type=int, default=256)

# **** Timbre data to Mel-generalized cepstrum arguments ***** #

# Order for Mel-generalized cepstrum analysis
parser.add_argument("--mcep_order", type=int, default=60)
# Alpha for Mel-generalized cepstrum analysis
parser.add_argument("--mcep_alpha", type=float, default=0.45)
# Gamma for Mel-generalized cepstrum analysis (synthesis)
parser.add_argument("--mcep_gamma", type=float, default=0)
# Input type for Mel-generalized cepstrum analysis
parser.add_argument("--mcep_input", type=int, default=3)
# Channels in the coded aperiodicity
parser.add_argument("--ap_channels", type=int, default=4)

# Constant to add such that the values are between them
parser.add_argument("--coding_const", type=float, default=0.5)

# ********** Noise removal arguments ************* #
# Noise remove arguments
parser.add_argument("--n_grad_freq", type=int, default=1)
parser.add_argument("--n_std_thresh", type=int, default=1)
parser.add_argument("--n_grad_time", type=int, default=1)

# Reverb level ( 0 means no reverb)
parser.add_argument("--reverb", type=int, default=30)

# FFT size
parser.add_argument("--fft_size", type=int, default=2048)


# **** Directories ***** #

# Directory of the training data
parser.add_argument("--training_dir", type=str, default="ProcessedData")
parser.add_argument("--data_dir", type=str, default="Dataset")
parser.add_argument("--model_dir", type=str, default="TrainedModels")
parser.add_argument("--output_dir", type=str, default="Output")

# ************ Model Parameters **************** #

# ******************* Harmonic model ****************** #
h_parser = argparse.ArgumentParser()
h_parser.add_argument("--start_pad", type=int, default=9)
h_parser.add_argument("--levels", type=int, default=3)
h_parser.add_argument("--blocks", type=int, default=2)
h_parser.add_argument("--dil_chan", type=int, default=130)
h_parser.add_argument("--res_chan", type=int, default=130)
h_parser.add_argument("--skip_chan", type=int, default=240)
h_parser.add_argument("--out_chan", type=int, default=240)
h_parser.add_argument("--output_chan", type=int, default=60)
h_parser.add_argument("--init_kernel", type=int, default=10)
h_parser.add_argument("--kernel", type=int, default=2)
h_parser.add_argument("--epochs", type=int, default=1000)
h_parser.add_argument("--l2_decay", type=float, default=0.00005)
h_parser.add_argument("--learn_rate", type=float, default=0.00005)
h_parser.add_argument("--temp", type=float, default=0.05)
h_parser.add_argument("--kernel_init", type=str, default="glorot_uniform")


# ******************* Aperiodic model ****************** #
a_parser = argparse.ArgumentParser()
a_parser.add_argument("--start_pad", type=int, default=9)
a_parser.add_argument("--levels", type=int, default=3)
a_parser.add_argument("--blocks", type=int, default=2)
a_parser.add_argument("--dil_chan", type=int, default=20)
a_parser.add_argument("--res_chan", type=int, default=20)
a_parser.add_argument("--skip_chan", type=int, default=16)
a_parser.add_argument("--out_chan", type=int, default=16)
a_parser.add_argument("--init_kernel", type=int, default=10)
a_parser.add_argument("--kernel", type=int, default=2)
a_parser.add_argument("--epochs", type=int, default=250)
a_parser.add_argument("--l2_decay", type=float, default=0.00005)
a_parser.add_argument("--learn_rate", type=float, default=0.000001)
a_parser.add_argument("--temp", type=float, default=0.01)
a_parser.add_argument("--kernel_init", type=str, default="glorot_uniform")

# ******************* Frequency model ****************** #
f_parser = argparse.ArgumentParser()
f_parser.add_argument("--start_pad", type=int, default=19)
f_parser.add_argument("--levels", type=int, default=7)
f_parser.add_argument("--blocks", type=int, default=2)
f_parser.add_argument("--dil_chan", type=int, default=100)
f_parser.add_argument("--res_chan", type=int, default=100)
f_parser.add_argument("--skip_chan", type=int, default=100)
f_parser.add_argument("--out_chan", type=int, default=4)
f_parser.add_argument("--init_kernel", type=int, default=20)
f_parser.add_argument("--kernel", type=int, default=2)
f_parser.add_argument("--epochs", type=int, default=300)
f_parser.add_argument("--l2_decay", type=float, default=0.00005)
f_parser.add_argument("--learn_rate", type=float, default=0.000001)
f_parser.add_argument("--temp", type=float, default=0.01)
f_parser.add_argument("--kernel_init", type=str, default="glorot_uniform")
