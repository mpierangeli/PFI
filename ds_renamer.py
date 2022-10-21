import os


fp = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\finales"
fp2 = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\sorted"
allfiles = os.listdir(fp2)
allfilesfinales = os.listdir(fp)
cont = int(len(allfilesfinales)/2)
cont2 = int(len(allfilesfinales)/2)
for file in sorted(allfiles):
    print(file)
    if ("segmentation" in file ) or ("Segmentation" in file): 
        cont+=1
        os.rename(os.path.join(fp2,file),os.path.join(fp,"segmentacion_"+str(cont)+".nii"))
    else:
        cont2+=1
        os.rename(os.path.join(fp2,file),os.path.join(fp,"sujeto_"+str(cont2)+".nii"))
