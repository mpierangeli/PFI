import os
import nibabel as nib
import numpy as np
import cv2
import torchio as tio
from torchio.transforms import HistogramStandardization

fp = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\aa"
fp2 = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\separadas"
allfiles = os.listdir(fp)
allfilesfinales = os.listdir(fp2)
cont = 0
cont2 = 0
sujetos = []
for file in sorted(allfiles):
    if "sujeto" in file:
        sujetos.append(os.path.join(fp,file))

landmarks = HistogramStandardization.train(sujetos)

transform = HistogramStandardization(landmarks)
