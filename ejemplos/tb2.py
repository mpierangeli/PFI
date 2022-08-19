import nibabel as nib
import matplotlib.pyplot as plt
img = nib.load("Case00.nii")
test = img.get_fdata()
print(img.header)

plt.imshow(test[:,:,15],cmap="gray")
plt.show()