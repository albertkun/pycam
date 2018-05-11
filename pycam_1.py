import numpy as np
#pip install opencv-python
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('alsdosa.png',1)


plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
#img2 = img[:,:,::-1] # flip the order to bgr, because opencv reads color in bgr not rbg order...

#alternative method to convert the colors
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.subplot(122);plt.imshow(img2) 
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

#cap = cv2.VideoCapture(0)

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Display the resulting frame
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()