import numpy as np
from flask import Flask, flash, request, jsonify, render_template
import pickle
import numpy as np
import os
import cv2
import tensorflow as tf
import time

app = Flask(__name__)
new_model = tf.keras.models.load_model('saved_model/my_model')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    url = [x for x in request.form.values()]
    final_features = [np.array(url)]
    
    print(final_features[0][0])
    # prediction = model.predict(final_features[0][0])
    video_name = final_features[0][0]
    count=0
    images = []
    vidObj = cv2.VideoCapture(video_name) 
    print("------------Request Logs--------------")
    print("Reading video data from link")
    total_fight = 0
    total_no_fight = 0
    total = 0
    output = ""
    while True: 
        print("Reading frames of video data")
        success, image = vidObj.read() 
        # Saves the frames with frame-count 
        if success:
            cv2.resize(image,(299,299))
            x = np.expand_dims(image, axis=0)
            images = np.vstack([x])
            classes = new_model.predict(images, batch_size=10)
            if classes[0][0]>0.5:
                total_fight += 1
            else:
                total_no_fight += 1
            total += 1
        else:
            break
    print("Predicted frames from model")
    vidObj.release() 
    cv2.destroyAllWindows() 
    if total_fight > total_no_fight:
        output = "Fight"
    else:
        output = "No Fight"
    print("Video is a "+ output + "video")
    return render_template('index.html', prediction_text='Video Type :  $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)