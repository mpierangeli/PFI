import os


fp = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\nuevas"
fp2 = r"C:\Users\Pier\Desktop\PFI_main\ds_mezcla_segmentacion integra\nuevas2"
allfiles = os.listdir(fp)
allfilesfinales = os.listdir(fp2)
cont = int(len(allfilesfinales)/2)
cont2 = int(len(allfilesfinales)/2)
for file in sorted(allfiles):
    print(file)
    if ("segmentation" in file ) or ("Segmentation" in file): 
        cont+=1
        os.rename(os.path.join(fp,file),os.path.join(fp2,"segmentacion_"+str(cont).zfill(3)+".nii"))
    else:
        cont2+=1
        os.rename(os.path.join(fp,file),os.path.join(fp2,"sujeto_"+str(cont2).zfill(3)+".nii"))
