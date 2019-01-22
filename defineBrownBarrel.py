import cv2
import numpy as np
import matplotlib.pyplot as plt
from roipoly import RoiPoly as ro
import pickle
import os

def display(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows
Brown_trainingData = [[],[],[]]
image_path = os.listdir('C:/Users/TonyY/AppData/Local/Programs/Python/Python37-32/code/276HW1/trainset')
for image in image_path:
    #Read image
    img1 = cv2.imread('.\\trainset\\'+image)
    img2 = cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
    #show image and define region of interest
    plt.imshow(img2, interpolation='nearest')
    plt.show(block=False)
    roi = ro(color='r')
    #return an np array where the region you draw is True and others are false
    mask = roi.get_mask(img2)
    a = np.where(mask == True)
    img_tcrcb = cv2.cvtColor(img1,cv2.COLOR_RGB2YCR_CB)


    for i in range(len(a[0])):
        Brown_trainingData[0].append(img_tcrcb[a[0][i],a[1][i],0])
        Brown_trainingData[1].append(img_tcrcb[a[0][i],a[1][i],1])
        Brown_trainingData[2].append(img_tcrcb[a[0][i],a[1][i],2])
file = open("BrownData","wb")
pickle.dump(Brown_trainingData,file)
file.close()
