import cv2
import numpy as np

# Load DNN model
modelFile = "dnn model/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "dnn model/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

camera = cv2.VideoCapture(0) # Open Camera
camera.set(3, 720) 
camera.set(4, 1280)

# Face Detection
def face_detection(frame):
    # Convert the frame to a blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), # Resize the frame
                                            1.0, (300, 300), 
                                            (104.0, 177.0, 123.0) # Normalize the frame
            
                                )
    net.setInput(blob)
    faces = net.forward()
    return faces

# Box around the face and draw a rectangle on it
def drawer_box(frame):
    faces = face_detection(frame)
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.8:
            box = faces[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, w, h) = box.astype("int")
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 4)

# Close  camera when 'q' is pressed
def close_window():
    camera.release()
    cv2.destroyAllWindows()
    exit()

# Run  the program until 'q' is pressed
def main():
    while True:
        _, frame = camera.read()
        frame = cv2.flip(frame, 1)
        drawer_box(frame)
        cv2.imshow("Face Detection", frame) # Showing camera app
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):   # Exit condition
            close_window()
            
if __name__ == '__main__':
    main()
