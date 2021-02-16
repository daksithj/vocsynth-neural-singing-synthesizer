import numpy as np
import pandas as pd
from math import log2, pow

from args import parser

params = parser.parse_args()

A4 = 440
C0 = A4 * pow(2, -4.75)
name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def extract_notes(f_data):
    notes = []
    for freq in f_data:
        if freq > 0:
            h = round(12 * log2(freq / C0))
            octave = h // 12
            n = h % 12
            notes.append(name[n] + str(octave))
        else:
            notes.append("None")
    notes = np.asarray(notes)
    return notes


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
        try:
            pre_note_array.append(notes[y-1])
        except IndexError:
            pre_note_array.append("sp")
        try:
            post_note_array.append(notes[y+1])
        except IndexError:
            post_note_array.append("sp")

    note_position_data = pd.DataFrame(notes, columns=['Note'])
    pre_note_position_data = pd.DataFrame(pre_note_array, columns=['Pre_note'])
    post_note_position_data = pd.DataFrame(post_note_array, columns=['Post_note'])
    note_timing_data = pd.DataFrame(timing_array, columns=['Note_timings'])
    note_data = pd.concat([note_position_data, note_timing_data, pre_note_position_data,
                              post_note_position_data], axis=1)

    note_data = pd.get_dummies(data=note_data, columns=['Note'], prefix='Note')
    note_data = pd.get_dummies(data=note_data, columns=['Pre_note'], prefix='Pre_note')
    note_data = pd.get_dummies(data=note_data, columns=['Post_note'], prefix='Post_note')
    note_data = pd.get_dummies(data=note_data, columns=['Note_timings'], prefix='Note_timings')

    note_data = np.asarray(note_data)
    return note_data

def process_frequency(f_data, label_data):
    f_labels = label_data[:, 256:]
    notes = extract_notes(f_data)
    note_data = get_note_data(notes)
    f_labels = np.concatenate([f_labels, note_data], axis=1)


