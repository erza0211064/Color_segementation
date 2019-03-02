import os
import pickle
import numpy as np
color_class = ["Blue","notBlue","Brown","Green"]
#color_class = ["Blue"]
color_data = {}
#load data
data_path = os.listdir('C:/Users/TonyY/AppData/Local/Programs/Python/Python37-32/code/276HW1/trainset')
for color in color_class:
    #read training data
    file = open(color+"Data","rb")
    allRGB_data = color_data[color] = pickle.load(file)
    file.close()
    #transform array to matrix for covariance
    allRGB_matrix = np.matrix(allRGB_data)
    color_shape = allRGB_matrix.shape
    #calculate mean
    mean1 = np.mean(color_data[color][0])
    mean2 = np.mean(color_data[color][1])
    mean3 = np.mean(color_data[color][2])
    mean = [[mean1],[mean2],[mean3]]
    #caluate covariance
    cov = sum([(allRGB_matrix[:,i] - mean) * (allRGB_matrix[:,i] - mean).T for i in range(color_shape[1])]) / (color_shape[1] - 1)
    print(color,":")
    print("n = ",len(color_data[color][0]))
    print("mean:",mean)
    print("covariance:",cov)


'''
Blue :
n =  397881
mean: [[82.98560624910463], [160.67991937287783], [96.14792362540558]]
covariance: [[1396.88205129  378.47902106 -579.45781603]
 [ 378.47902106  285.80364042 -282.30556282]
 [-579.45781603 -282.30556282  404.24315826]]
notBlue :
n =  538286
mean: [[94.9431268879369], [158.51298937739418], [99.06022077482973]]
covariance: [[1415.90816554  187.09603399 -321.80242697]
 [ 187.09603399  275.65843127 -241.81672732]
 [-321.80242697 -241.81672732  318.79453799]]
Brown :
n =  796325
mean: [[92.00199667221298], [146.54363356669703], [111.57899601293441]]
covariance: [[1360.60516677  205.00316281 -299.55720737]
 [ 205.00316281  503.81020933 -496.1783063 ]
 [-299.55720737 -496.1783063   568.74278546]]
Green :
n =  922093
mean: [[87.92265530700266], [142.39764535681326], [113.50468445156834]]
covariance: [[1388.34667566  266.07137972 -309.06704581]
 [ 266.07137972  557.6467478  -483.80989818]
 [-309.06704581 -483.80989818  526.61898923]]
'''
