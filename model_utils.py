import numpy as np
import tensorflow as tf
from keras import backend as K
from args import parser

params = parser.parse_args()


def split_layer(layer, parts, axis):
    split = tf.split(layer, parts, axis=axis)
    return split


def temp_calc(args, temp):
    param_factor = 4

    means, sigmas, weights = args

    if temp == 0.05:
        temps = []

        for i in range(params.mcep_order):
            if i <= 3:
                temperature = 0.05
            elif i >= 8:
                temperature = 0.5
            else:
                temperature = 0.05 + (i - 3) * 0.09

            temps.append(temperature)

        temps = tf.convert_to_tensor(temps, dtype='float64')
        temps = K.reshape(temps, K.shape(means[0]))
        temp_factor = K.sqrt(temps)
    else:
        temps = temp
        temp_factor = np.sqrt(temps)

    alt_means = 0

    for k in range(param_factor):
        alt_means = alt_means + weights[k] * means[k]

    for k in range(param_factor):
        means[k] = means[k] + (alt_means - means[k]) * (1 - temps)
        sigmas[k] = sigmas[k] * temp_factor

    return means, sigmas, weights


def multi_params(output, temp):

    param_factor = 4
    r_u = 1.6
    r_s = 1.1
    r_w = 1 / 1.75

    output = K.expand_dims(output, axis=-1)
    output = K.reshape(output, shape=(K.shape(output)[0], K.shape(output)[1], -1, param_factor))

    split1 = output[:, :, :, 0]
    split2 = output[:, :, :, 1]
    split3 = output[:, :, :, 2]
    split4 = output[:, :, :, 3]

    xi = 2 * K.sigmoid(split1) - 1
    omega = K.exp(4 * K.sigmoid(split2)) * 2 / 255
    alpha = 2 * K.sigmoid(split3) - 1
    beta = 2 * K.sigmoid(split4)

    sigmas = []

    for k in range(param_factor):
        sigma = omega * K.exp(k * (K.abs(alpha) * r_s - 1))
        sigmas.append(sigma)

    means = []
    for k in range(param_factor):
        t_sum = 0
        for l in range(k):
            t_sum = t_sum + (sigmas[l] * r_u * alpha)
        mean = xi + t_sum
        means.append(mean)

    weights = []
    t_sum = 0

    for k in range(param_factor):
        t_sum = t_sum + (K.pow(alpha, (2*k)) * K.pow(beta, k) * (r_w ** k))

    for k in range(param_factor):
        weight = (K.pow(alpha, (2*k)) * K.pow(beta, k) * (r_w ** k)) / t_sum
        weights.append(weight)

    if temp != 0:
        means, sigmas, weights = temp_calc([means, sigmas, weights], temp)

    return means, sigmas, weights


def network_loss(target, output):

    means, sigmas, weights = multi_params(output, 0)
    probability = 0
    for k in range(4):
        variance = (sigmas[k] ** 2)

        sig_trace = K.log(sigmas[k])
        pi_log = np.log(np.sqrt(2 * np.pi))
        log_prob = -((target - means[k]) ** 2) / (2 * variance) - sig_trace - pi_log

        probability = probability + weights[k] * log_prob

    return -K.mean(probability)

