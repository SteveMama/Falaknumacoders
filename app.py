from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import cv2
import numpy as np
import heapq

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.wsgi import WSGIServer
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/Inception_model_f.h5'

#Load your trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary to make everything ready to run on the GPU ahead of time
print('Model loaded. Start serving...')

disease_name = ["Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule","Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema","Fibrosis","Pleural Thickening","Hernia"]

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = cv2.imread(img_path)
    width = int(img.shape[1] * 0.250)
    height = int(img.shape[0] * 0.250)
    sample_image2 = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    sample_image2 = sample_image2.reshape((1, 256, 256, 3))
    print(sample_image2.shape)
    # Preprocessing the image
    #img = image.img_to_array(img)
    #img = np.expand_dims(img, axis=0)


    preds = model.predict(sample_image2)

    return preds


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

        # Make prediction
        preds = model_predict(file_path, model)
        os.remove(file_path)#removes file from the server after prediction has been returned

        # Arrange the correct return according to the model. 
		# In this model 1 is Pn eumonia and 0 is Normal.

        def disease(arr):
            ind = heapq.nlargest(6, range(len(arr)), arr.take)
            str1 = f"{disease_name[ind[0]]} Probability:{arr[ind[0]]}"
            str2 = f"{disease_name[ind[1]]} Probability:{arr[ind[1]]}"
            str3 = f"{disease_name[ind[2]]} Probability:{arr[ind[2]]}"
            str4 = f"{disease_name[ind[3]]} Probability:{arr[ind[3]]}"
            str5 = f"{disease_name[ind[4]]} Probability:{arr[ind[4]]}"
            str6 = f"{disease_name[ind[5]]} Probability:{arr[ind[5]]}"

            str7 = ','.join([str1, str2, str3,str4,str5,str6])
            # print("Disease:", disease_name[index], "Probability:", max_val)
            return str(str7)

        return str(disease(preds[0]))

    return None

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

    #this section is used by gunicorn to serve the app on Heroku
if __name__ == '__main__':
        app.run()
    #uncomment this section to serve the app locally with gevent at:  http://localhost:5000
    # Serve the app with gevent 
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
