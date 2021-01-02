
from imutils.video import VideoStream
import numpy as np
from imutils.video import FPS
import imutils
import time
import cv2
from keras.models import load_model


CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('helmet_deploy.prototxt.txt', 'helmet_deploy.caffemodel')

print('Loading helmet model...')
loaded_model = load_model('new_helmet_model.h5')
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print("[INFO] starting video stream...")


cap = cv2.VideoCapture('vid4.mp4') 


fps = FPS().start()


while True:
	# i = not i
	# if i==True:

    try:
        
        
        ret, frame = cap.read()

        
        frame = imutils.resize(frame, width=600, height=600)

        
        (h, w) = frame.shape[:2]
        
        
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        
        net.setInput(blob)

        detections = net.forward()  
        
        persons = []
        person_roi = []
        motorbi = []
        
        
        for i in np.arange(0, detections.shape[2]):
            
            confidence = detections[0, 0, i, 2]
            
            
            
            if confidence > 0.5:
                
                
                idx = int(detections[0, 0, i, 1])
                
                if idx == 15:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    # roi = box[startX:endX, startY:endY/4] 
                    # person_roi.append(roi)
                    persons.append((startX, startY, endX, endY))

                if idx == 14:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    motorbi.append((startX, startY, endX, endY))

        xsdiff = 0
        xediff = 0
        ysdiff = 0
        yediff = 0
        p = ()
        
        for i in motorbi:
            mi = float("Inf")
            for j in range(len(persons)):
                xsdiff = abs(i[0] - persons[j][0])
                xediff = abs(i[2] - persons[j][2])
                ysdiff = abs(i[1] - persons[j][1])
                yediff = abs(i[3] - persons[j][3])

                if (xsdiff+xediff+ysdiff+yediff) < mi:
                    mi = xsdiff+xediff+ysdiff+yediff
                    p = persons[j]
                    


            if len(p) != 0:

	            
	            label = "{}".format(CLASSES[14])
	            print("[INFO] {}".format(label))
	            cv2.rectangle(frame, (i[0], i[1]), (i[2], i[3]), COLORS[14], 2)
	            y = i[1] - 15 if i[1] - 15 > 15 else i[1] + 15
	            cv2.putText(frame, label, (i[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[14], 2)   
	            label = "{}".format(CLASSES[15])
	            print("[INFO] {}".format(label))

	            cv2.rectangle(frame, (p[0], p[1]), (p[2], p[3]), COLORS[15], 2)
	            y = p[1] - 15 if p[1] - 15 > 15 else p[1] + 15

	            roi = frame[p[1]:p[1]+(p[3]-p[1])//4, p[0]:p[2]]
	            print(roi)
	            if len(roi) != 0:
	            	img_array = cv2.resize(roi, (50,50))
	            	gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
	            	img = np.array(gray_img).reshape(1, 50, 50, 1)
	            	img = img/255.0
	            	prediction = loaded_model.predict_proba([img])
	            	cv2.rectangle(frame, (p[0], p[1]), (p[0]+(p[2]-p[0]), p[1]+(p[3]-p[1])//4), COLORS[0], 2)
	            	cv2.putText(frame, str(round(prediction[0][0],2)), (p[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0], 2)

    except:
        pass

    cv2.imshow('Frame', frame)  
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'): 
        break
     
    
    fps.update()
	    


fps.stop()

print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
cv2.destroyAllWindows()
cap.release()   
