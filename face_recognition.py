import cv2
import numpy as np
import os 
from keras.models import load_model

# Load the model
model = load_model('model/face_recognition.h5')

reverse_label_dict = np.load('model/user_dataset.npy', allow_pickle=True).item()
user_names = {v: k for k, v in reverse_label_dict.items()}

# Load DNN model
modelFile = "dnn model/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "dnn model/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 720) # set video width
cam.set(4, 1280) # set video height

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Downsize the image to speed up processing
    small_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
    
    blob = cv2.dnn.blobFromImage(small_img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.8:  # Adjusted confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([small_img.shape[1], small_img.shape[0], small_img.shape[1], small_img.shape[0]])
            (x, y, w, h) = box.astype("int")
            x *= 2  # Upscale the bounding box
            y *= 2
            w *= 2
            h *= 2
            cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
            face_roi = gray[y:y+h, x:x+w]
            
            # Preprocess the face image for the model
            face_roi = cv2.resize(face_roi, (224, 224))  # Adjust size to match the model's input size
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)  # Convert to RGB
            face_roi = np.expand_dims(face_roi, axis=0)  # Add batch dimension
            face_roi = face_roi / 255.0  # Normalize pixel values
            
            # Make prediction using the model
            confidence_val = model.predict(face_roi)
            id = np.argmax(confidence_val)
            confidence_percent = confidence_val[0][id] * 100
            
            id = user_names[id]

            cv2.putText(
                img, 
                str(id), 
                (x+5, y-5), 
                font, 
                1, 
                (255, 255, 255), 
                2
            )
            
            # Calculate text size to adjust the position
            textSize = cv2.getTextSize("{:.2f}%".format(confidence_percent), font, 1, 2)[0]
            text_x = x + 5
            text_y = h + 25

            cv2.putText(
                img, 
                "{:.2f}%".format(confidence_percent), 
                (text_x, text_y), 
                font, 
                1, 
                (255, 255, 0), 
                1
            )  
    
    cv2.imshow('camera', img) 
    k = cv2.waitKey(10) & 0xff # Press 'q' for exiting video
    if k == ord("q"):
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
