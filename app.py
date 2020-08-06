
from __future__ import division, print_function
import os

import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__, static_url_path='/static')

# Model saved with Keras model.save()
MODEL_PATH = 'imageprediction.h5'
classifier = load_model(MODEL_PATH)

# this is key : save the graph after loading the model
global graph
graph = tf.get_default_graph()

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('food_dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
            
ch = [[1,0,0,0]]
cu = [[0,1,0,0]]
ho = [[0,0,1,0]]
pi = [[0,0,0,1]]


#density - gram / cm^3
density_dict = { 1:0.599, 2:0.609, 3:0.94, 4:0.577}
#kcal
calorie_dict = { 1:97, 2:52, 3:89, 4:92}
#skin of photo to real multiplier
skin_multiplier = 5*2.3

def getCalorie(label, volume): #volume in cm^3
    	
    calorie = calorie_dict[int(label)]
    if (volume == None):
        print("aaa")
        return None, None , calorie
    density = density_dict[int(label)]
    mass = volume*density*1.0
    calorie_tot = (calorie/100.0)*mass
    #return mass, calorie_tot, calorie #calorie per 100 grams
    return calorie_tot #calorie per 100 grams
    

def getVolume(label, fruit_area, mask_fruit2, fruit_final, skin_area, fruit_contour, pix_to_cm_multiplier):
	'''
	Using callibration techniques, the volume of the food item is calculated using the
	area and contour of the foot item by comparing the foot item to standard geometric shapes
	'''
	area_fruit = (fruit_area/skin_area)*skin_multiplier #area in cm^2
	label = int(label)
	volume = 100
    
    
	if label == 1 : #chickpizza
		volume = area_fruit*0.5
		 
        
	if label == 2 : #sphere-cupcake
		radius = np.sqrt(area_fruit/np.pi)
		volume = (4/3)*np.pi*radius*radius*radius
		print (area_fruit, radius, volume, skin_area)
		

	if label == 3 : #cylinder hotdog
		fruit_rect = cv2.minAreaRect(fruit_contour)
		height = max(fruit_rect[1])*pix_to_cm_multiplier
		radius = 0.5*(np.sqrt(height*height +(2*area_fruit/np.pi)) - height)
		volume = np.pi*radius*radius*height

		

	if label == 4 : #vegpizza
		volume = area_fruit*2

        
	
	return volume

def getAreaOfFood(img1):
	img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	img_filt = cv2.medianBlur( img, 5)
	img_th = cv2.adaptiveThreshold(img_filt,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	# find contours. sort. and find the biggest contour. the biggest contour corresponds to the plate and fruit.
	mask = np.zeros(img.shape, np.uint8)
	largest_areas = sorted(contours, key=cv2.contourArea)
	cv2.drawContours(mask, [largest_areas[-1]], 0, (255,255,255,255), -1)
	img_bigcontour = cv2.bitwise_and(img1,img1,mask = mask)

	# convert to hsv. otsu threshold in s to remove plate
	hsv_img = cv2.cvtColor(img_bigcontour, cv2.COLOR_BGR2HSV)
	h,s,v = cv2.split(hsv_img)
	mask_plate = cv2.inRange(hsv_img, np.array([0,0,100]), np.array([255,90,255]))
	mask_not_plate = cv2.bitwise_not(mask_plate)
	fruit_skin = cv2.bitwise_and(img_bigcontour,img_bigcontour,mask = mask_not_plate)

	#convert to hsv to detect and remove skin pixels
	hsv_img = cv2.cvtColor(fruit_skin, cv2.COLOR_BGR2HSV)
	skin = cv2.inRange(hsv_img, np.array([0,10,60]), np.array([10,160,255])) #Scalar(0, 10, 60), Scalar(20, 150, 255)
	not_skin = cv2.bitwise_not(skin); #invert skin and black
	fruit = cv2.bitwise_and(fruit_skin,fruit_skin,mask = not_skin) #get only fruit pixels

	fruit_bw = cv2.cvtColor(fruit, cv2.COLOR_BGR2GRAY)
	fruit_bin = cv2.inRange(fruit_bw, 10, 255) #binary of fruit

	#erode before finding contours
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	erode_fruit = cv2.erode(fruit_bin,kernel,iterations = 1)

	#find largest contour since that will be the fruit
	img_th = cv2.adaptiveThreshold(erode_fruit,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	mask_fruit = np.zeros(fruit_bin.shape, np.uint8)
	largest_areas = sorted(contours, key=cv2.contourArea)
	cv2.drawContours(mask_fruit, [largest_areas[-2]], 0, (255,255,255), -1)
	#dilate now
	kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
	mask_fruit2 = cv2.dilate(mask_fruit,kernel2,iterations = 1)
	res = cv2.bitwise_and(fruit_bin,fruit_bin,mask = mask_fruit2)
	fruit_final = cv2.bitwise_and(img1,img1,mask = mask_fruit2)
	#find area of fruit
	img_th = cv2.adaptiveThreshold(mask_fruit2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	largest_areas = sorted(contours, key=cv2.contourArea)
	fruit_contour = largest_areas[-2]
	fruit_area = cv2.contourArea(fruit_contour)

	
	#finding the area of skin. find area of biggest contour
	skin2 = skin - mask_fruit2
	#erode before finding contours
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	skin_e = cv2.erode(skin2,kernel,iterations = 1)
	img_th = cv2.adaptiveThreshold(skin_e,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	mask_skin = np.zeros(skin.shape, np.uint8)
	largest_areas = sorted(contours, key=cv2.contourArea)
	cv2.drawContours(mask_skin, [largest_areas[-2]], 0, (255,255,255), -1)

	skin_rect = cv2.minAreaRect(largest_areas[-2])
	box = cv2.boxPoints(skin_rect)
	box = np.int0(box)
	mask_skin2 = np.zeros(skin.shape, np.uint8)
	cv2.drawContours(mask_skin2,[box],0,(255,255,255), -1)

	pix_height = max(skin_rect[1])
	pix_to_cm_multiplier = 5.0/pix_height
	skin_area = cv2.contourArea(box)

	return fruit_area, mask_fruit2, fruit_final, skin_area, fruit_contour, pix_to_cm_multiplier


def model_predict(file_path, classifier):
    
    test_image = image.load_img(file_path, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                               shear_range = 0.2,
                                               zoom_range = 0.2,
                                               horizontal_flip = True)

    training_set = train_datagen.flow_from_directory('food_dataset/training_set',
                                                     target_size = (64, 64),
                                                     batch_size = 32,
                                                     class_mode = 'binary')
            
            
            
    result = result.astype (int)
    print (result)
    result = result.tolist()
    print (result)
            
    training_set.class_indices
    
    if(result==ch):
        label = 1
        pred = 'Chicken pizza has '
            
    elif(result==cu):
        label = 2
        pred = 'Cupcake has '
        
    elif(result==ho):
        label = 3
        pred = 'hot dog has '
            
    elif (result==pi):
        label = 4
        pred = 'Veg pizza has '
                #print ("Food predicted is Pizza!")
            
    else:
        pred = 'No image predicted'
         
    img1 = cv2.imread(file_path)
    a,b,c,d,e,f= getAreaOfFood(img1)
        
    y=getVolume(label,a,b,c,d,e,f)
    x=getCalorie(label,y)
    
    x = str(x) + ' calories'
    
    print ("Calorie of ",pred,"is ",x)

                
    return pred + x
    


    


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')




@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        img1 = cv2.imread(file_path)
        a,b,c,d,e,f= getAreaOfFood(img1)


        #result = classifier.predict(test_image)
        
        with graph.as_default():
            result = model_predict(file_path, classifier)
            
            return result
    
            
        return None
                
            

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=8080, type=int, help='port to listen on')
    args = parser.parse_args()
    port = args.port

    app.run(host='127.0.0.1', port=port)
