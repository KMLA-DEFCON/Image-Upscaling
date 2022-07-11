import cv2
import numpy as np



image_scale = 0



img = cv2.imread("./dog.png") 
size = img.shape
dimension = (size[0], size[1])
new_dimension = (int(dimension[1]*2), int(dimension[0]*2))
resized_img = cv2.resize(img, new_dimension, interpolation=cv2.INTER_NEAREST)
cv2.imwrite("filename.png", resized_img)

    

