import csv
import os

import soundfile as sf
from tqdm import tqdm
import pandas as pd
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import sys
import numpy as np

train_manifest='/home/fbmao/Datasets/ccmixter_corpus/manifest.csv'
train_dataset = '/home/fbmao/Datasets/ccmixter_corpus'

'''
(Dir Structure)
dataset
  |-song1
     |- mixture.wav
     |- vocals.wav
     |- instrumental.wav
  |-song2
  	 |- mixture.wav
	 |- vocals.wav
	 |- instrumental.wav

'''

'''
def read_manifest(dataset, start = 0):
    rows = []
    with open(MANIFEST_DIR.format(dataset), 'r') as f:
        reader = csv.reader(f)
        for sid, aid, filename, duration, samplerate in reader:
            rows.append([int(sid) + start, aid, filename, duration, samplerate])
            n_speakers = int(sid) + 1
    return n_speakers, rows
'''

def save_manifest(train_manifest, rows):
    rows.sort()
    with open(train_manifest, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows)



def create_manifest(train_dataset):
    n_speakers = 0
    log = []
    
    for song in tqdm(os.listdir(train_dataset), desc = None):
        song_dir = os.path.join(train_dataset, song)
        if os.path.isdir(song_dir) == False:
            continue
        for audio in os.listdir(song_dir):
            audio_path = os.path.join(song_dir, audio)
            
            if 'mix' in audio:
                mix_path = audio_path
            elif 'vocal' in audio:
                vocal_path = audio_path
            elif 'instru' in audio:
                instru_path = audio_path
            else:
              continue

            
            filename = os.path.join(song_dir, audio)
            info = sf.info(filename)
            duration = info.duration
            samplerate = info.samplerate
        log.append((mix_path, vocal_path, instru_path, duration, samplerate))
    save_manifest(train_manifest, log)

'''
def merge_manifest(datasets, dataset):
    rows = []
    n = len(datasets)
    start = 0
    for i in range(n):
        n_speakers, temp = read_manifest(datasets[i], start = start)
        rows.extend(temp)
        start += n_speakers
    with open(MANIFEST_DIR.format(dataset), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def cal_eer(y_true, y_pred):
    fpr, tpr, thresholds= roc_curve(y_true, y_pred, pos_label = 1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh

'''

if __name__ == '__main__':
    create_manifest(train_dataset)
