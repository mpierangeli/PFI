import pydicom
# import pydicom_seg
# import SimpleITK as sitk

# dcm = pydicom.dcmread('1-1.dcm')

# reader = pydicom_seg.SegmentReader()
# result = reader.read(dcm)

# for segment_number in result.available_segments:
#     image_data = result.segment_data(segment_number)  # directly available
#     image = result.segment_image(segment_number)  # lazy construction
#     sitk.WriteImage(image, f'segmentation-{segment_number}.nrrd', True)

import numpy as np
import nrrd
import matplotlib.pyplot as plt

# Some sample numpy data
filename = 'segmentation-4.nrrd'

# Read the data back from file
readdata, header = nrrd.read(filename)
plt.ion()

while(1):
    for i in range(9):
        dcm = pydicom.dcmread("1-0"+str(i+1)+".dcm")
        plt.imshow(np.flipud(np.rot90(readdata[:,:,18-i],1))*1000+dcm.pixel_array)
        plt.title(str(i))
        plt.show()
        plt.pause(0.5)
        plt.clf()
    for i in range(9,19):
        dcm = pydicom.dcmread("1-"+str(i+1)+".dcm")
        plt.imshow(np.flipud(np.rot90(readdata[:,:,18-i],1))*1000+dcm.pixel_array)
        plt.title(str(i))
        plt.show()
        plt.pause(0.5)
        plt.clf()