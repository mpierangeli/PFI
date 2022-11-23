import os
import nibabel as nib
import numpy as np

fp = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\sec_ordenadas\sujetos"
fp2 = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\sec_ordenadas\sujetos"
allfiles = os.listdir(fp)
for file in sorted(allfiles):
    print(file)
    x = nib.load(os.path.join(fp,file))
    img = x.get_fdata()/32767.0
    nifti = nib.Nifti1Image(img, header = x.header, affine= x.affine)
    nib.save(nifti, os.path.join(fp2, file))
