Copy right by TUNG-LIN, YANG / t2yang@eng.ucsd.edu

ECE 276A Project 1 Color Segmentation

-barrel_detector.py

A BarrelDetector class for testing data on gradescope. Include segment_image and get_bounding_box function
segment_image: get the result of classified image
get_bounding_box: receive the (x1, y1),(x2,y2) of the bounding box

Main function include saving result image programming.

-Gaussian.py

Define "simple gaussian" function, "gaussian mixture model". 
Define "load_data" to load training data for gaussian mixture model.
Define "Evaluate" function include mean and covariance of every class for simplw gaussian model. 
Also provide a flag, mode, to change model Flag = 0 for simple gaussian and Flag = 1 for gaussian mixture model. this function will return a mask of segmentation and an matrix show the class of each pixel.

-defineTrainData.py

Load in every image from trainset and using "roipoly" to extract training data. Save the data into .pkl file

-caluate_mean_covariance.py

Load training data and caluate mean and covariance for each class.

-class_evaluate.py

Define a class evaluate, include every function above:
load_data - read training data
load_image - read test images
get_prior - caluate the prior probability of each class
gaussian - gaussian equation
simple_gaussian - simple gaussian model and maximum likelihood estimation
detectImg - do the whole draw bounding box processing, including read data, classify and draw bounding boxes
draw_bounding_box - draw bounding box using regionporps function
print_parameter - get training data info, debug use

Goal of class_evaluate.py is to do every process of drawing bounding boxes on a test image automatically as long as we have training data.
