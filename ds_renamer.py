import nrrd
import os


fp = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\todas"
fp2 = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\sorted"
allfiles = os.listdir(fp)
cont = 0
cont2 = 0
for file in sorted(allfiles):
    print(file)
    if ("segmentation" in file ) or ("Segmentation" in file): 
        cont+=1
        os.rename(os.path.join(fp,file),os.path.join(fp2,"seg_"+str(cont)+".nii"))
    else:
        cont2+=1
        os.rename(os.path.join(fp,file),os.path.join(fp2,"sujeto_"+str(cont2)+".nii"))
