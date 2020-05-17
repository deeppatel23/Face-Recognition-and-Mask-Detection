import cv2
#full code is similar to above explained in above face recognition module

detector = cv2.CascadeClassifier('C:/Users/Dell/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/Dell/anaconda3/Lib/site-packages/cv2/data/haarcascade_eye.xml')
#this classifier to detect faces and eye respectively
cap = cv2.VideoCapture(0)

while (True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)#rectangle around face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray,minNeighbors=15)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 255), 2)#rectangle around eyes

    if faces is ():
        cv2.putText(img, "MASK Detected - ENTRY ALLOWED", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2) 
    else:
        cv2.putText(img, "NO MASK - PLEASE WEAR MASK", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2) 

    cv2.imshow('frame', img)
    if cv2.waitKey(1)==13:
        break
    #process ends when you press enter

cap.release()
cv2.destroyAllWindows()