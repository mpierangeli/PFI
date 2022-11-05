import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import shutil

fp = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\aa"
fp2 = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\flipped"
allfiles = os.listdir(fp)
for file in sorted(allfiles):
    if "sujeto" in file:
        x = nib.load(os.path.join(fp,file))
        img = x.get_fdata()
        inputImage = sitk.ReadImage(os.path.join(fp,file))
        maskImage = sitk.OtsuThreshold(inputImage,0,1,200)
        inputImage = sitk.Cast(inputImage,sitk.sitkFloat32)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        output = corrector.Execute(inputImage,maskImage)
        newimg = np.transpose(sitk.GetArrayViewFromImage(output))
        nifti = nib.Nifti1Image(newimg, header = x.header, affine= x.affine)
        nib.save(nifti, os.path.join(fp2, file))  
    else:
        shutil.copy2(os.path.join(fp,file), os.path.join(fp2,file))