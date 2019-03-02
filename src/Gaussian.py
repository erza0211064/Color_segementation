import numpy as np
import matplotlib.pyplot as plt
import math
import os, cv2
import pickle

#load data
def load_data():
    #add the name of each class
    color_class = ["Blue","notBlue","Brown","Green"]
    color_data = {}
    for color in color_class:    
        #load data
        file = open(color+"Data","rb")
        allRGB_data = color_data[color] = pickle.load(file)
        file.close()
        #transform array to matrix for covariance
        allRGB_matrix = np.matrix(allRGB_data)
        color_shape = allRGB_matrix.shape
    return color_data
#gaussian mixture model
def gmm_likelihood(x, mean, cov):
    k, m = mean.shape
    likelihood = 0
    for i in range(k):
        likelihood += simple_gaussian(x, mean[i], cov[i] * np.eye(m))
    return likelihood/k
#gaussian eqaution  
def simple_gaussian(x,mean,cov):
    """
    #Do gaussian for each pixel(slow)
    a = math.sqrt(((2*math.pi) **3))
    b = math.sqrt(abs(np.linalg.det(cov)))
    #a = (2*math.pi)**(3/2)
    #b = abs(np.linalg.det(cov))**2
    c = (x-mean).T
    d = np.linalg.inv(cov)
    e = x-mean
    return math.exp( (-0.5) * c * d * e ) / ( a * b )
    """
    #Do gaussian in matrix(fast)
    cov_inv = np.linalg.cholesky(np.linalg.inv(cov))
    a = np.exp((-0.5) * np.sum(np.square(np.dot((x-mean).T,cov_inv)), axis = 1))
    b = np.sqrt(((2 * math.pi) ** 3) * (np.linalg.det(cov)))
    return a/b
#EM algorithm
def gaussian_mixture(x,k):
    n,m = x.shape
    #inital mean and covariance
    cur_u = np.random.uniform(low=0,high=1,size=(k,m))*255
    sigma = np.ones((k, m)) * 1000
    prior_prob = np.ones((n,k))/k
    thresh = 1
    for i in range(30):
        pre_u = np.copy(cur_u)
        #E step
        for j in range(k):
            print(cur_u[j,:].shape)
            prior_prob[:,j] = gmm_likelihood(x.T, cur_u[j,:], sigma[j]*np.eye(m))
        prior_prob /= sum(prior_prob.T)
        #M step
        for j in range(k):
            prior_prob_reshape = prior_prob[:,j].reshape((n,1))
            cur_u[j] = prior_prob_reshape.T*x/sum(prior_prob_reshape)
            sigma[j] = prior_prob_reshape.T*((x-cur_u[j])**2) / sum(prior_prob_reshape)
        print(cur_u)
        if np.linalg.norm(cur_u-pre_u)<thresh:
            break
    return cur_u, sigma
#classify the test image
def evaluate(img, title):
    #mode = 0: single gaussian ; mode = 1: gaussian mixture 
    mode = 0
    
    #prior probability
    n_blue = 352480
    n_notblue = 183575
    n_brown = 232232
    n_green = 88482
        
    p_blue = n_blue /(n_blue + n_notblue + n_brown + n_green)
    p_notblue = n_notblue /(n_blue + n_notblue + n_brown + n_green)
    p_brown = n_brown /(n_blue + n_notblue + n_brown + n_green)
    p_green = n_green /(n_blue + n_notblue + n_brown + n_green)
    
    #mean and covariance
    blue_mean =  [[86.30065819337267], [163.45427541988198], [93.62571777122106]]
    blue_cov = [[1214.69627254,  291.19626335, -521.34939405],
        [ 291.19626335,  227.27805361, -231.39182989],
        [-521.34939405, -231.39182989,  371.15547045]]
    notblue_mean = [[118.15798992237505], [150.6481764946207], [107.1322075446003]]
    notblue_cov = [[940.47524605, -24.29925841, -67.98969579],
        [-24.29925841, 172.48966154, -63.39900663],
        [-67.98969579, -63.39900663,  41.80629786]]

    brown_mean = [[83.6684737676117], [120.04783147886596], [140.11057477005753]]
    brown_cov = [[869.80238841, -10.53194624,  36.40607956],
        [-10.53194624,  50.83237246, -52.20074092],
        [ 36.40607956, -52.20074092,  70.94725317]]
    green_mean = [[63.13912434167401], [116.87720666350218], [123.54686828959562]]
    green_cov = [[ 8.48335003e+02, -1.36593432e+02, -1.29044073e-01],
        [-1.36593432e+02,  1.01617826e+02, -2.19487040e+01],
        [-1.29044073e-01, -2.19487040e+01,  8.38901475e+01]]

    #shape of test image
    nx, ny, nz = img.shape 
    #change image into 2d matrix to do gaussian, x is input
    x = np.zeros((3, nx*ny))
    x[0] = np.reshape(img[:,:,0],(1, -1))
    x[1] = np.reshape(img[:,:,1],(1, -1))
    x[2] = np.reshape(img[:,:,2],(1, -1))
    if mode == 0:
        #apply MLE
        py_blue = simple_gaussian(x, blue_mean, blue_cov)
        py_notblue = simple_gaussian(x, notblue_mean, notblue_cov)
        py_brown = simple_gaussian(x, brown_mean, brown_cov)
        py_green = simple_gaussian(x, green_mean, green_cov)
        #combine four image and find out each pixel's class
        combineImg = np.vstack((py_blue, py_notblue, py_brown, py_green)).T
        resImg = np.argmax(combineImg, axis=1)
        class_array = resImg
        resImg[resImg != 0] = 1
        resImg[resImg == 0] = 100
        resImg[resImg == 1] = 0
        resImg[resImg == 100] = 255
        #reshape result
        res = np.reshape(resImg, (nx, ny))
    if mode == 1:
        EM_blue_mean, EM_blue_cov = gaussian_mixture(x,8)
        EM_notblue_mean, EM_notblue_cov = gaussian_mixture(x,8)
        EM_brown_mean, EM_brown_cov = gaussian_mixture(x,8)
        EM_green_mean, EM_green_cov = gaussian_mixture(x,8)
        
        py_blue = simple_gaussian(x, blue_mean, blue_cov)
        py_notblue = simple_gaussian(x, notblue_mean, notblue_cov)
        py_brown = simple_gaussian(x, brown_mean, brown_cov)
        py_green = simple_gaussian(x, green_mean, green_cov)
        #combine four image and find out each pixel's class
        combineImg = np.vstack((py_blue, py_notblue, py_brown, py_green)).T
        resImg = np.argmax(combineImg, axis=1)
        class_array = resImg
        resImg[resImg != 0] = 1
        resImg[resImg == 0] = 100
        resImg[resImg == 1] = 0
        resImg[resImg == 100] = 255
        #reshape result
        res = np.reshape(resImg, (nx, ny))
 
    '''
    #check the result
    plt.imshow(res)
    plt.title(title)
    plt.show()
    '''
    return res, class_array
