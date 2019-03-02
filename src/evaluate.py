import cv2
import numpy as np
import pickle
import os, math
from Gaussian import simple_gaussian, evaluate
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

def display(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows

image_path = os.listdir('C:/Users/TonyY/AppData/Local/Programs/Python/Python37-32/code/276HW1/validset')
for f in os.listdir('./validset/'):
    img_test = cv2.imread('./validset/'+f)
    img_test_ycrcb = cv2.cvtColor(img_test, cv2.COLOR_RGB2YCR_CB)
    image_array, class_array = evaluate(img_test_ycrcb, f)
    #plt.imshow(image_array)
    #plt.show()
    #cv2.waitKey(0)
    image_array = np.array(image_array, dtype='uint8')
    
    contours, hierarchy = cv2.findContours(image_array, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        continue
    
    Max = np.max([cv2.contourArea(c) for c in contours])
    allaxis = []
    center = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 300:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if h/w < 1.0 or h/w > 3:
            continue
        #print(h/w)
        allaxis.append([x,y,w,h])
        center.append([int((2*x+w)/2),int((2*y+h)/2)])
    '''
    #caluate the distence between center
    left = 0
    right = left + 1
    maxD = math.sqrt((center[left][0]-center[right][0])**2 + (center[left][1]-center[right][1])**2)
    for i in range(1,len(center)-1):
        left = i
        right = left + 1
        tmp = math.sqrt((center[left][0]-center[right][0])**2 + (center[left][1]-center[right][1])**2)
        if tmp > maxD:
            maxD = tmp
    '''
    for i in range(len(center)-1):
        a = cv2.line(img_test, (center[i][0],center[i][1]), (center[i+1][0],center[i+1][1]),(0, 255, 0), 2) 
    for i in allaxis:
        a = cv2.rectangle(img_test, (i[0],i[1]),(i[0]+i[2], i[1]+i[3]), (0, 255, 0), 2)
    print(f)
    print(center)
    #cv2.imshow('title',a)
    #cv2.waitKey(0)

    cv2.imwrite('C:/Users/TonyY/Desktop/All stuff/sensing & estimation in robotics/HW/testfigure/' +f, img_test)
    '''
    rect = cv2.minAreaRect(c)
    bbox = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
    '''
        
    '''
    #regionprops
    label_img = label(image_array)

    regions = regionprops(label_img)
    regions.pop(0)
    wantRegion = regions[1]
    for i in regions:
        if i.area > wantRegion.area:
            wantRegion = i

    #get bounding box
    img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
    plt.imshow(img_test)

    minr, minc, maxr, maxc = wantRegion.bbox

    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)

    plt.plot(bx, by, '-r')
    plt.title(f)
    plt.show()
    
    cv2.imwrite('C:/Users/TonyY/Desktop/All stuff/sensing & estimation in robotics/HW/testfigure/' +f, plt.figure())
    '''

    
