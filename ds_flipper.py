import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

fp = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\finales2"
fp2 = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\flipped"
allfiles = os.listdir(fp)
for file in sorted(allfiles):
    x = nib.load(os.path.join(fp,file))
    img = x.get_fdata()
    img = np.rot90(img,3)
    nifti = nib.Nifti1Image(img, header = x.header, affine= x.affine)
    nib.save(nifti, os.path.join(fp2, file))  
