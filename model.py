import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.layers import Input, Lambda, Conv1D, ZeroPadding1D, Add, Activation, Multiply
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN
from keras import backend as K
from model_utils import multi_params, split_layer, network_loss
from data_handler import HarmonicDataSet
import args


class SingingModel:

    def __init__(self,
                 spectral_data,
                 aperiodic_data,
                 label_data,
                 cutoff_points,
                 name):

        self.harmonic_data_set = HarmonicDataSet(spectral_data, aperiodic_data, label_data, cutoff_points)

        self.name = name

        model_dir = args.parser.parse_args().model_dir

        if not os.path.isdir(model_dir + '/'):
            os.mkdir(model_dir)

        self.model_path = model_dir + '/' + name

        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)

    @staticmethod
    def sample_output(output, temp):
        output = output[:, -1, :]
        output = K.expand_dims(output, axis=1)

        means, sigmas, weights = multi_params(output, temp)
        out = 0
        for k in range(4):
            out += means[k] * weights[k]

        spectral_value = K.zeros_like(means[0])

        random = K.random_uniform(K.shape(weights[0]), minval=0.0, maxval=1.0)

        for k in range(4):
            mask_set = K.zeros_like(weights[k])
            for i in range(k):
                mask_set = mask_set + weights[i]
            mask_a = K.less(random, (weights[k] + mask_set))
            mask_b = K.greater_equal(random, mask_set)
            mask = tf.math.logical_and(mask_a, mask_b)
            mask = K.cast(mask, dtype='float64')
            distribution = K.random_normal(K.shape(means[k]), mean=means[k], stddev=sigmas[k])
            spectral_value = spectral_value + (mask * distribution)
        return spectral_value

    @staticmethod
    def build_model(data_handler, m_params):

        K.set_floatx('float64')

        levels = m_params.levels
        blocks = m_params.blocks
        dil_chan = m_params.dil_chan
        res_chan = m_params.res_chan
        skip_chan = m_params.skip_chan
        initial_kernel = m_params.init_kernel
        kernel_size = m_params.kernel
        l2_decay = m_params.l2_decay
        kernel_init = m_params.kernel_init

        input_chan = data_handler.get_data_channels()
        cond_chan = data_handler.get_label_channels()

        start_pad = m_params.start_pad

        # Inputs
        input_layer = Input(shape=(None, input_chan,))
        label_input = Input(shape=(None, cond_chan,))

        # Padding the data
        x = ZeroPadding1D((start_pad, 0), name='start_pad')(input_layer)
        x = Conv1D(res_chan, initial_kernel, use_bias=True, kernel_initializer=kernel_init, name='start-conv',
                   kernel_regularizer=l2(l2_decay))(x)

        skip_connection = 0

        for i in range(blocks):
            neural_levels = levels
            dilation_rate = 1
            if i == (blocks - 1):
                neural_levels = levels - 1
            for j in range(neural_levels):

                layer_name = "_Block_" + str(i) + "_Level_" + str(j)

                residual_input = x

                label_conv = Conv1D(dil_chan * 2, 1, use_bias=True, kernel_initializer=kernel_init,
                                    name=('label_conv' + layer_name), kernel_regularizer=l2(l2_decay))(label_input)

                x = Conv1D(dil_chan * 2, kernel_size, padding="causal", dilation_rate=dilation_rate, use_bias=True,
                           kernel_initializer=kernel_init, name=('dilated_conv' + layer_name),
                           kernel_regularizer=l2(l2_decay))(x)

                x = Add(name=('add_label_dil' + layer_name))([x, label_conv])

                filter_layer, gate_layer = Lambda(split_layer, arguments={'parts': 2, 'axis': 2},
                                                  name=('split' + layer_name))(x)

                filter_layer = Activation('tanh', name=('filter' + layer_name))(filter_layer)
                gate_layer = Activation('sigmoid', name=('gate' + layer_name))(gate_layer)

                x = Multiply(name=('multiply' + layer_name))([filter_layer, gate_layer])

                skip_layer = x

                skip_layer = Conv1D(skip_chan, 1, use_bias=True, kernel_initializer=kernel_init,
                                    name=('skip' + layer_name), kernel_regularizer=l2(l2_decay))(skip_layer)

                try:
                    skip_connection = Add(name=('add_skip' + layer_name))([skip_layer, skip_connection])
                except:
                    skip_connection = skip_layer

                x = Conv1D(res_chan, 1, use_bias=True, kernel_initializer=kernel_init,
                           name=('residual' + layer_name), kernel_regularizer=l2(l2_decay))(x)

                x = Add(name=('add_res' + layer_name))([x, residual_input])

                dilation_rate = dilation_rate * 2

        label_output = Conv1D(skip_chan, 1, use_bias=True, kernel_initializer=kernel_init,
                              name='label_out', kernel_regularizer=l2(l2_decay))(label_input)

        x = Add(name='add_skip_out')([skip_connection, label_output])

        x = Activation("tanh", name='output')(x)

        x = Conv1D(skip_chan, 1, use_bias=True, kernel_initializer=kernel_init, name='final-output',
                   kernel_regularizer=l2(l2_decay))(x)

        network = Model([input_layer, label_input], x)

        return network

    @staticmethod
    def lr_scheduler(epoch, lr):
        return lr / (1 + 0.00001 * epoch)

    def train_model(self, model_type, load=True, epochs=0):

        data_handler = self.harmonic_data_set.set_type(model_type)

        if model_type == 0:
            m_params = args.h_parser.parse_args()
            model_loc = self.model_path + '/harmonic_model.h5'
        else:
            m_params = args.a_parser.parse_args()
            model_loc = self.model_path + '/aperiodic_model.h5'

        if epochs == 0:
            epochs = m_params.epochs

        if load and not os.path.isfile(model_loc):
            print('Cannot find model :' + model_loc + "\n Creating new model...")
            load = False

        if load:
            model = load_model(model_loc)
            print('Successfully loaded :' + model_loc + "\n Continuing training...")
        else:
            model = self.build_model(data_handler, m_params)

            adam_optimizer = Adam(learning_rate=m_params.learn_rate)
            model.compile(optimizer=adam_optimizer, loss=network_loss)

        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)

        lr_schedule = LearningRateScheduler(self.lr_scheduler)
        nan_terminator = TerminateOnNaN()
        checkpoint = ModelCheckpoint(model_loc, monitor='loss', verbose=1, save_best_only=True, mode='min')

        model.fit(data_handler, epochs=epochs, callbacks=[nan_terminator, lr_schedule, checkpoint])

    def get_generator(self, model_type):

        if model_type == 0:
            model_loc = self.model_path + '/harmonic_model.h5'
            temp = args.h_parser.parse_args().temp
        else:
            model_loc = self.model_path + '/aperiodic_model.h5'
            temp = args.a_parser.parse_args().temp

        if not os.path.isfile(model_loc):
            sys.exit('Cannot find model :' + model_loc)
        else:
            K.set_floatx('float64')
            model = load_model(model_loc, compile=False)
            sample_layer = Lambda(self.sample_output, name="sample_layer", arguments={'temp': temp})(model.output)

            generator = Model(model.input, sample_layer)
            return generator

    @staticmethod
    def get_receptive_field(model_type):

        if model_type == 0:
            m_params = args.h_parser.parse_args()
        else:
            m_params = args.a_parser.parse_args()

        receptive_field = 1

        for i in range(m_params.blocks):
            neural_levels = m_params.levels
            add_scope = m_params.kernel - 1

            if i == (m_params.blocks - 1):
                neural_levels = m_params.levels - 1

            for j in range(neural_levels):
                receptive_field = receptive_field + add_scope
                add_scope = add_scope * 2

        receptive_field = receptive_field + m_params.init_kernel - 1

        return receptive_field

    def inference(self, label_data, model_type, spectral_data=None):

        generator = self.get_generator(model_type)

        receptive_field = self.get_receptive_field(model_type)

        init_channels = args.parser.parse_args().mcep_order
        if model_type == 0:
            channels = init_channels
        else:
            channels = args.parser.parse_args().ap_channels
            init_channels += channels

        audio_length = label_data.shape[0]
        model_input = np.zeros((1, 1, init_channels))
        output = np.zeros((audio_length, channels))

        for i in tqdm(range(audio_length)):

            if i < receptive_field:
                start = 0
                end = i + 1
            else:
                start = i + 1 - receptive_field
                end = i + 1

            labels = label_data[start:end, :]
            labels = np.expand_dims(labels, axis=0)

            generated = generator.predict([np.array(model_input), np.array(labels)])
            generated = np.squeeze(generated, axis=0)

            output[i, :] = generated

            if i < receptive_field - 1:
                model_input = output[:i + 1, :]
                if spectral_data is not None:
                    spectral_output = spectral_data[:i + 1, :]
                    model_input = np.concatenate([model_input, spectral_output], axis=1)
                model_input = np.pad(model_input, ((1, 0), (0, 0)))
            else:
                model_input = output[i + 1 - receptive_field: i + 1, :]
                if spectral_data is not None:
                    spectral_output = spectral_data[i + 1 - receptive_field: i + 1, :]
                    model_input = np.concatenate([model_input, spectral_output], axis=1)

            model_input = np.expand_dims(model_input, axis=0)

        return output
