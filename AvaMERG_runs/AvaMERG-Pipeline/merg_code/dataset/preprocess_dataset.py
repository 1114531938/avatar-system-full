import json
import os.path
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import re
import random
import numpy as np
import torch
import os
from typing import Literal

def standardized_rename(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            dia_num = filename[3:8].lstrip('0')  
            utt_num = str(int(filename[11]) + 1) 
            new_filename = f'dia{dia_num}utt{utt_num}.wav'
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
            print(f'Renamed: {filename} -> {new_filename}')

        elif filename.endswith('.mp4'):
            dia_num = filename[3:8].lstrip('0')  
            utt_num = str(int(filename[11]) + 1)  
            new_filename = f'dia{dia_num}utt{utt_num}.mp4'
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
            print(f'Renamed: {filename} -> {new_filename}')

if __name__ == '__main__':
    original_audio_path = 'merg_data/video'
    standardized_rename(original_audio_path)


  
