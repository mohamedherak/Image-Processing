import cv2
from skimage import io, measure
import matplotlib.pyplot as plt
import numpy as np

img = io.imread('marbles.jpg')

hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

mask = cv2.inRange(hsv, (100,90,90), (130,255,255)) #blue mask

from scipy import ndimage as nd

closed_mask= nd.binary_closing(mask, np.ones((11,11)))

label_image = measure.label(closed_mask)

from skimage.color import label2rgb

image_label_overlay = label2rgb(label_image, image =img)

plt.subplot(141),plt.imshow(mask)
plt.xticks([]), plt.yticks([])
#plt.title('Mask'), plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(closed_mask)
plt.xticks([]), plt.yticks([])
#plt.title('Binary closing'), plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(label_image)
plt.xticks([]), plt.yticks([])
#plt.title('Image Original'), plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(image_label_overlay)
#plt.title('Image Original'), 
plt.xticks([]), plt.yticks([])
plt.show()

props = measure.regionprops_table(label_image,img, properties = ['label','area','equivalent_diameter','mean_intensity','solidity'])

import pandas as pd

df =pd.DataFrame(props)
print(df.head())
