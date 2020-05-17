import cv2
#this is to import library opencv
#OpenCV, the CV is an abbreviation form of a computer vision
#OpenCV is a Python open-source library, which is used for computer vision in Artificial intelligence, Machine Learning, face recognition, etc.

face_classifier = cv2.CascadeClassifier('C:/Users/Dell/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
#classifier - It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images. Here we will work with face detection.
#above is location in my pc of classifier(frontalface_default is trained to detect faces)

def face_extractor(img):
#thsi function extracts the image from the face
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Using cv2.COLOR_BGR2GRAY color space for convert BGR image to grayscale  
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    #detecting your face

    if faces is():
        return None #means if no faces is found then return face 

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
        #cropping the only required face area

    return cropped_face


cap = cv2.VideoCapture(0)
#opening your camera
count = 0

while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face = cv2.resize(face_extractor(frame),(200,200))
        #resize your face
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_name_path = 'C:/Users/Dell/Desktop/facesimage/user'+str(count)+'.jpg'
        #file name and its path to store your images captured
        cv2.imwrite(file_name_path,face)
        #saving the image

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        #writing text on image
        cv2.imshow('Face Cropper',face)
        #dispalys the resulting frame
    else:
        print("Face not Found")
        pass

    if cv2.waitKey(1)==13 or count==200:
        break
    #either the process will stop after capturing 200 pictures or when you press enter

cap.release()
cv2.destroyAllWindows()
#above mention two steps are most important to free your camera from your code
print('Colleting Samples Complete!!!')