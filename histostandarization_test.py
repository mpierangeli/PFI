import torch
import numpy as np
from pathlib import Path
import torchio as tio
from torchio.transforms import HistogramStandardization
import os

fp = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\sec_ordenadas\sujetos"
fp2 = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\test_mhd"
allfiles = os.listdir(fp)

lista = [os.path.join(fp,file) for file in allfiles]

# t1_landmarks_path = Path('t1_landmarks.npy')


# t1_landmarks = (
#     t1_landmarks_path
#     if t1_landmarks_path.is_file()
#     else HistogramStandardization.train(lista)
# )
# torch.save(t1_landmarks, t1_landmarks_path)



landmarks_dict = {
     't1': "t1_landmarks.npy"
}

torch.save(landmarks_dict, 'path_to_landmarks.pth')
transform = tio.HistogramStandardization('path_to_landmarks.pth')

out = transform(lista)