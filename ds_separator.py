import os
import nibabel as nib
import numpy as np
import cv2
import torchio as tio


fp = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\flipped"
fp2 = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\separadas2"
allfiles = os.listdir(fp)
allfilesfinales = os.listdir(fp2)
cont = 0
cont2 = 0
rescale = tio.RescaleIntensity(out_min_max=(0, 255))
for file in sorted(allfiles):
    x = nib.load(os.path.join(fp,file))
    if "sujeto" in file:
        normalized = rescale(x)
        for slices in range(np.size(normalized,2)):
            cont+=1
            img = normalized.get_fdata()[:,:,slices]
            cv2.imwrite(fp2+"\img_sujeto_"+str(cont).zfill(4)+".png", img)
    else:
        seg = x.get_fdata()
        for slices in range(np.size(seg,2)):
            cont2+=1
            img = seg[:,:,slices]*255
            cv2.imwrite(fp2+"\img_segmentacion_"+str(cont2).zfill(4)+".png", img)
