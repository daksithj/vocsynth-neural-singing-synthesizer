import numpy as np
import pandas as pd
from math import log2, pow

a_4 = 440
c_0 = a_4 * pow(2, -4.75)
name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NULL_KEY = 'N'


def extract_notes(f_data):
    min_freq = 1000
    max_freq = -1
    notes = []
    for freq in f_data:
        if freq > 0:
            h = round(12 * log2(freq / c_0))
            notes.append(h)
            if h > max_freq:
                max_freq = h
            if h < min_freq:
                min_freq = h
        else:
            notes.append(-1)
    notes = np.asarray(notes)
    return notes, min_freq, max_freq


def get_note_data(notes):
    audio_length = notes.shape[0]

    x = 0
    timing_array = []

    while x < audio_length:
        note = notes[x]
        counter = 0
        while x + counter < audio_length and notes[x + counter] == note:
            counter = counter + 1

        for numerator in range(1, counter+1):

            timing = numerator / counter
            if timing <= 0.333:
                timing_array.append(0)
            elif timing <= 0.666:
                timing_array.append(1)
            else:
                timing_array.append(2)
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
    notes = np.expand_dims(notes, axis=1)
    pre_note_array = np.asarray(pre_note_array)
    pre_note_array = np.expand_dims(pre_note_array, axis=1)
    post_note_array = np.asarray(post_note_array)
    post_note_array = np.expand_dims(post_note_array, axis=1)

    note_data = np.concatenate([notes, pre_note_array, post_note_array], axis=1)

    note_timing_data = pd.DataFrame(timing_array, columns=['Note_timings'])
    note_timing_data = pd.get_dummies(data=note_timing_data, columns=['Note_timings'], prefix='Note_timings')
    note_timings = np.asarray(note_timing_data)

    return note_data, note_timings


def shift_pitch(note_data, frequency_data, p_shift):
    note_data = note_data + p_shift
    note_data[note_data == (p_shift - 1)] = -1

    f_shift = p_shift / 12
    frequency_data = frequency_data * pow(2, f_shift)

    return note_data, frequency_data


def notes_to_number(n_data):
    notes = []
    for key in n_data:
        if key == NULL_KEY:
            notes.append(-1)
        else:
            key = list(key)
            notes.append(int(key[1])*12 + name.index(key[0]))

    return notes


def note_to_frequency(notes):
    note_f = []
    for note in notes:
        if note > -1:
            freq = pow(2, note / 12) * c_0
            note_f.append(freq)
        else:
            note_f.append(0)

    return note_f


def smooth_out(original_frequency, gen_frequency, smooth_factor=20):

    notes, _, _ = extract_notes(original_frequency)

    note_f = note_to_frequency(notes)

    frequency = []

    for note, freq in zip(note_f, gen_frequency):
        dif = freq - note
        frequency.append(note + (dif/smooth_factor))

    return np.asarray(frequency)
