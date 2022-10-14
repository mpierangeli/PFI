import nibabel as nib
import matplotlib.pyplot as plt
img = nib.load("ejemplos/Case01_segmentation.nii")
imgm = nib.load("ejemplos/Case01.nii")
test = img.get_fdata()
testm = imgm.get_fdata()
#print(img.header)

plt.imshow(test[:,:,15]*testm[:,:,15],cmap="gray")
plt.show()