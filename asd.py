from PIL import Image
import numpy as np
import cv2

abc = np.asarray([1000,120,254])
cv2.imwrite("aa.tiff",abc)
im_frame = cv2.imread('aa.tiff')
print(im_frame)