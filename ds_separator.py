import os
import nibabel as nib
import numpy as np
import cv2
import torchio as tio



fp = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\bias_corrected_sec\mascaras"
fp2 = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\test_mhd"
allfiles = os.listdir(fp)
allfilesfinales = os.listdir(fp2)
cont = 0
cont2 = 0
rescale = tio.RescaleIntensity(out_min_max=(0, 1),percentiles=(1,99))

media = 354
var = 148
maxx = 22842.541015625

for file in sorted(allfiles):
    x = nib.load(os.path.join(fp,file))
    y = x.get_fdata()
    if "sujeto" in file:
        #normalized = rescale(x)
        for slices in range(np.size(y,2)):
            cont+=1
            #img = (y[63:319,63:319,slices]/2116.231308269967)*255 #num media de intensidad  max
            #img = (y[63:319,63:319,slices]-media)/var
            #img = y[63:319,63:319,slices]/maxx
            img = y[63:319,63:319,slices]
            img = rescale(np.reshape(img,(1,img.shape[0],img.shape[1],1)))
            np.save(fp2+"\sujetos\img_sujeto_"+str(cont).zfill(4),img[0,:,:,0])
            #cv2.imwrite(fp2+"\sujetos\img_sujeto_"+str(cont).zfill(4)+".png", img)
    else:
        for slices in range(np.size(y,2)):
            cont2+=1
            #img = y[63:319,63:319,slices]*255
            img = y[63:319,63:319,slices]
            np.save(fp2+"\mascaras\img_segmentacion_"+str(cont2).zfill(4),img)
            #cv2.imwrite(fp2+"\mascaras\img_segmentacion_"+str(cont2).zfill(4)+".png", img)
    
