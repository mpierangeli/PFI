import os
import nibabel as nib
import numpy as np
import cv2
import elasticdeform



fp = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\bias_corrected_sec"
fp2 = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\test_mhd"
fp3 = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\test_mhd\sujetos"
allfiles = os.listdir(fp)
allfilesfinales = os.listdir(fp2)
fpp = os.listdir(fp3)
cont = len(fpp)


for file in sorted(allfiles):
    if "sujeto" in file:
        x = nib.load(os.path.join(fp,file))
        y = x.get_fdata()
        for slice in range(np.size(y,2)):
            if np.random.randint(0,100) > 50:
                a = nib.load(os.path.join(fp,"segmentacion_"+file[-7:]))
                b = a.get_fdata()
                cont+=1
                img_deformed,seg_deformed = elasticdeform.deform_random_grid([y[:,:,slice],b[:,:,slice]], sigma=5, points=10)
                img = (img_deformed[63:319,63:319]/2116.231308269967)*255 
                seg = (seg_deformed[63:319,63:319])*255 
                cv2.imwrite(fp2+"\sujetos\img_sujeto_"+str(cont).zfill(4)+".png", img)
                cv2.imwrite(fp2+"\mascaras\img_segmentacion_"+str(cont).zfill(4)+".png", seg)
    
