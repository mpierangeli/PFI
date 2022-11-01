import os
from PIL import Image
import nibabel as nib
import numpy as np


fp = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\aa"
fp2 = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\separadas"
allfiles = os.listdir(fp)
allfilesfinales = os.listdir(fp2)
cont = 0
cont2 = 0
for file in sorted(allfiles):
    if "sujeto" in file:
        cont+=1
        x = nib.load(os.path.join(fp,file))
        img = x.get_fdata()
        for slices in range(np.size(img,2)):
            img = (np.asarray(img[:,:,slices])/np.asarray(img[:,:,slices]).max())*255
            slice = Image.fromarray(img)
            slice.save(fp2,"img_sujeto_"+str(cont).zfill(4)+".png")
    else:
        cont2+=1
        x = nib.load(os.path.join(fp,file))
        img = x.get_fdata()
        for slices in range(np.size(img,2)):
            img = (np.asarray(img[:,:,slices])/np.asarray(img[:,:,slices]).max())*255
            slice = Image.fromarray(img)
            slice.save(fp2+"img_segmentacion_"+str(cont2).zfill(4)+".png")
