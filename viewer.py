import nibabel as nib
import numpy as np
import os

fp2 = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\sorted"
allfiles = os.listdir(fp2)
slices = []
for file in allfiles:

    img = nib.load(os.path.join(fp2,file))
    test = img.get_fdata()
    slices.append(np.size(test,2))
print(slices)