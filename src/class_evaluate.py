import numpy as np
import matplotlib.pyplot as plt
import math
import os, cv2
import pickle
from skimage.measure import label, regionprops

class evaluate:
    def __init__(self):
        self.img_RGB = {}   #image in RGB
        self.img_YCrCb = {} #image in YCrCb
        self.img_res = {}   #image shows only blue
        self.img_class = {} #image shows the class of each pixel
        self.color_class = ["Blue","notBlue","Brown","Green"]
        self.color_mean = {'Blue': [[84.82227790360477], [162.41810956120483], [94.66648233600745]], \
                           'notBlue': [[110.11881911297583], [152.60152423660773], [106.80992905392016]], \
                           'Brown': [[92.00199667221298], [146.54363356669703], [111.57899601293441]], \
                           'Green': [[87.92265530700266], [142.39764535681326], [113.50468445156834]]}
        self.color_cov = {'Blue': [[1274.30766435,  336.30061412, -538.57907422],[ 336.30061412,  256.77535896, -255.81381285], [-538.57907422, -255.81381285,  378.08222077]], \
                          'notBlue': [[ 972.93687621,   86.518402  , -106.81730778], [  86.518402  ,  151.73303552,  -62.19430975], [-106.81730778,  -62.19430975,   46.14984547]], \
                          'Brown': [[1360.60516677,  205.00316281, -299.55720737], [ 205.00316281,  503.81020933, -496.1783063 ], [-299.55720737, -496.1783063 ,  568.74278546]], \
                          'Green': [[1388.34667566,  266.07137972, -309.06704581], [ 266.07137972,  557.6467478 , -483.80989818], [-309.06704581, -483.80989818,  526.61898923]]}
        self.color_n = {'Blue': 377944, 'notBlue': 176613, 'Brown': 796325, 'Green': 922093}
        self.total_train_data = sum([self.color_n[i] for i in self.color_n])

    #load training data    
    def load_data(self):
        print("loading data...")
        self.total_train_data = 0
        for color in self.color_class:
            color_data = {}
            #load data
            file = open(color+"Data","rb")
            allRGB_data = color_data[color] = pickle.load(file)
            file.close()
            #transform array to matrix for covariance
            allRGB_matrix = np.matrix(allRGB_data)
            color_shape = allRGB_matrix.shape
            #calculate mean
            mean1 = sum(color_data[color][0]) / len(color_data[color][0])
            mean2 = sum(color_data[color][1]) / len(color_data[color][1])
            mean3 = sum(color_data[color][2]) / len(color_data[color][2])
            mean = [[mean1],[mean2],[mean3]]
            #caluate covariance
            cov = sum([(allRGB_matrix[:,i] - mean) * (allRGB_matrix[:,i] - mean).T for i in range(color_shape[1])]) / (color_shape[1] - 1)
            #save n,mean,cov
            self.color_mean[color] = mean
            self.color_cov[color] = cov
            self.color_n[color] = len(color_data[color][0])
            #save total train data
            self.total_train_data += len(color_data[color][0])
        print("data loading complete!!")

    #load testing image
    def load_image(self):
        print("loading image...")
        for f in os.listdir('validset/'):
            tmp = cv2.imread("./validset/"+f)
            self.img_RGB[f] = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
            self.img_YCrCb[f] = cv2.cvtColor(tmp, cv2.COLOR_RGB2YCR_CB)
        
    def get_prior(self, color):
        return self.color_n[color]/self.total_train_data

    def gaussian(self, x,mean,cov):
        cov_inv = np.linalg.cholesky(np.linalg.inv(cov))
        a = np.exp((-0.5) * np.sum(np.square(np.dot((x-mean).T,cov_inv)), axis = 1))
        b = np.sqrt(((2 * math.pi) ** 3) * (np.linalg.det(cov)))
        return a/b
    def simple_gaussian(self):
        for imgIdx in self.img_YCrCb:                            
            img = self.img_YCrCb
            #shape of test image
            nx, ny, nz = img[imgIdx].shape 
            #change image into 2d matrix to do gaussian, x is input
            x = np.zeros((3, nx*ny))
            x[0] = np.reshape(img[imgIdx][:,:,0],(1, -1))
            x[1] = np.reshape(img[imgIdx][:,:,1],(1, -1))
            x[2] = np.reshape(img[imgIdx][:,:,2],(1, -1))
            #apply MLE
            py_blue = self.gaussian(x, self.color_mean['Blue'], self.color_cov['Blue']) 
            py_notblue = self.gaussian(x, self.color_mean['notBlue'], self.color_cov['notBlue'])
            py_brown = self.gaussian(x, self.color_mean['Brown'], self.color_cov['Brown'])
            py_green = self.gaussian(x, self.color_mean['Green'], self.color_cov['Green'])
            #combine four image and find out each pixel's class
            combineImg = np.vstack((py_blue, py_notblue, py_brown, py_green)).T
            resImg = np.argmax(combineImg, axis=1)
            #save to image class dictionary
            self.img_class[imgIdx] = resImg
            #turn blue prediction into color and other into black
            resImg[resImg != 0] = 1
            resImg[resImg == 0] = 100
            resImg[resImg == 1] = 0
            resImg[resImg == 100] = 1
            print(resImg.shape)
            #reshape result
            res = np.reshape(resImg, (nx, ny))
            '''
            #check the result
            plt.imshow(res)
            plt.title(imgIdx)
            plt.show()
            '''
            #save to image result dictionary
            self.img_res[imgIdx] = res
    
        
    def detectImg(self, mode):
        print("detect image...")
        p_blue = self.get_prior('Blue')
        p_notblue = self.get_prior('notBlue')
        p_brown = self.get_prior('Brown')
        p_green = self.get_prior('Green')
        #classify
        if mode == 0:
            self.simple_gaussian()
        if mode == 1:
            self.gaussian_mixture()
        self.draw_bounding_box()
        print("finish!!")

    def draw_bounding_box(self):
        print("drawing bounding box...")
        
        mode = 0
        if mode == 0:
            #using regionprops method
            for imgIdx in self.img_YCrCb:
                label_img = label(self.img_res[imgIdx])
     
                regions = regionprops(label_img)
                regions.pop(0)
                wantRegion = regions[1]
                for i in regions:
                    if i.area > wantRegion.area:
                        wantRegion = i

                #get bounding box
                plt.imshow(self.img_RGB[imgIdx])

                minr, minc, maxr, maxc = wantRegion.bbox

                bx = (minc, maxc, maxc, minc, minc)
                by = (minr, minr, maxr, maxr, minr)

                plt.plot(bx, by, '-r')
                plt.show()
        if mode == 1:
            #using find contour
            for imgIdx in self.img_YCrCb:
                imageDilate = np.zeros((800, 1200, 3))
                imageDilate[:,:,0] = self.img_res[imgIdx]
                imageDilate[:,:,1] = self.img_res[imgIdx]
                imageDilate[:,:,2] = self.img_res[imgIdx]
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
                imageDilate = cv2.erode(imageDilate, kernel, iterations = 1)
                imageDilate = cv2.dilate(imageDilate, kernel, iterations = 3)
                #change type to uint8
                imageDilate = np.array(imageDilate, dtype='uint8')
                _, thresh = cv2.threshold(imageDilate, 0, 255, 0)
                #change 3d to 2d to implement findContours
                thresh = cv2.cvtColor(thresh, cv2.COLOR_RGB2GRAY)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    
                #image_array = np.array(image_array, dtype='uint8')
                #contours, hierarchy = cv2.findContours(image_array, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                contourList = [cv2.contourArea(c) for c in contours]
                #avoid exception if no contour is found
                if contourList == []:
                        MaxContour = 1
                else:
                        MaxContour = contours[contourList.index(max(contourList))]
                for c in contours:
                        area = cv2.contourArea(c)
                        #get rid of two small area
                        if area < cv2.contourArea(MaxContour) * 0.4:
                            continue
                        x, y, w, h = cv2.boundingRect(c)
                        #get rid of not barrel like contour
                        if h/w < 1.4 or h/w > 2.9:
                            continue
                        #save the axis
                        boxes.append([x,y,x+w, y+h])    
    #for debug

    def print_parameter(self):
        print("image in Y CR CB:",self.img_YCrCb.keys())
        print("number of training data:", self.color_n)
        print("mean of each color class:", self.color_mean)
        print("cov of each color class:", self.color_cov)


                
#data_path = os.listdir('C:/Users/TonyY/AppData/Local/Programs/Python/Python37-32/code/276HW1/')
c1 = evaluate()
c1.load_image()
c1.load_data()
c1.print_parameter()
c1.detectImg(0)
#print(c1.get_prior('Blue'))

