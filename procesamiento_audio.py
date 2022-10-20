import os
import librosa
import pandas as pd
import numpy as np
import random


def espectrograma_mel_archivo(file):
    sample_rate = librosa.get_samplerate(file)
    duration = librosa.get_duration(filename=file, sr=sample_rate)
    offset = 0
    if (duration > 20):
      max_offset = int(duration - 20)
      max_offset = max_offset - 1 if max_offset > 0 else 0
      duration = 20
      # print(f'Max offset: {max_offset} - duration: {duration} - offset: {offset}')
      offset = random.randint(0, max_offset)
    
    # print(f'Duration: {duration} - offset: {offset}')

    # print(f'*** Sample rate: {sample_rate} - duration: {duration} ***')
    y, sr = librosa.core.load(file, duration=duration, sr=sample_rate, offset=offset)
    # print(f'*** y: {y} - sr: {sr}')

    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=1024)
    S_db = librosa.power_to_db(S, ref=np.max)

    if S_db.shape[1] != 657:
        S_db.resize(128, 657, refcheck=False)

    return S_db


def espectrogramas_mel():
    labels = []
    mel_specs = []

    directorios = ["blues", "classical", "country", "rock", "jazz"]

    for directorio in directorios:
        actual = f"./db/{directorio}"

        for file in os.scandir(actual):
            labels.append(directorio)
            S_db = espectrograma_mel_archivo(file)
            mel_specs.append(S_db)

    X = np.array(mel_specs)

    labels = pd.Series(labels)
    label_dict = {"blues": 0, "classical": 1, "country": 2, "rock": 3, "jazz": 4}
    y = labels.map(label_dict).values

    return X, y


def espectrograma_mel(path):
    mel_specs = []
    S_db = espectrograma_mel_archivo(path)
    mel_specs.append(S_db)
    X = np.array(mel_specs)
    return X
