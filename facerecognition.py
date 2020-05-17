import cv2
import numpy as np #necessary in training model
from os import listdir
from os.path import isfile, join
#above libraries are necessary to import files(where you previously saved your photos) to the code

data_path = 'C:/Users/Dell/Desktop/facesimage/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
#gathering files where  you previously save your images

Training_Data, Labels = [], []
#creating empty lists

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #reading the main image
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    #using numpy for training code
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()
#The Local Binary Pattern Histogram(LBPH) algorithm is a simple solution on face recognition problem, which can recognize both front face and side face

model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Model Training Complete!!!!!")

face_classifier = cv2.CascadeClassifier('C:/Users/Dell/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
#location of cascade classifier

def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2) #drawing rectangle around your face if face is found
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))
        #roi - Many common image operations are performed using Region of Interest in OpenCV. A ROI allows us to operate on a rectangular subset of the image. The typical series of steps to use ROI is: create a ROI on the image, perform the operation you want on this subregion of the image, reset back the ROI 

    return img,roi

cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()

    image, face = face_detector(frame)

    cv2.putText(image, "FACE RECOGNITION!!!", (150, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)
        #predict is inbuilt function gives the result of % of match found


        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300)) #confidence is calculated in precentage only
            display_string = str(confidence)+'% Confidence it is user'
        cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)


        if confidence > 80:
            #the values of confidence is need to keep above 75% for accuracy
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', image)

        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)


    except:
        cv2.putText(image, "Face Not Found", (150, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass

    if cv2.waitKey(1)==13:
        break
    #the process ends when you press enter


cap.release()
cv2.destroyAllWindows()