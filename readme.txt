Hello Guys!!!

Today, I will tell you how even you can make face recognition project and mask detection project via opencv python library on your own.

To further help combat the coronavirus, I decided to open up face mask detection technology to the public for free. This quick detection system can be widely used in travel scenes, with mobile phone photos, surveillance images etc., and is able to work round the clock.

Prerequisites:
1.) Python environment is required for the code to run (eg. Anaconda3).
2.) Install following libraries using command prompt – opencv, numpy, opencv-contrib-python (these libraries are not found in python module thus need to be installed externally).
3.) Basic py3 knowledge is necessary.
4.) You much know the use and functions on opencv library.
5.) Make sure light of room is always on for better results!!!


Steps to make facial recognition project:
1.) Firstly, you have to create a folder consisting your images as we need a stored data.
2.) Refer to file facedatacollection.py, further I will explain the whole code in comments itself.
3.)  After collecting the data of your photos in a folder, now you have to train your model and it’s ready for face detection.
4.) Refer to facerecognition.py code, I have explained the whole code in comments over there & HURRAY!! Our project Is ready.
5.) Once you understand face recognition, mask detection is quiet simple project.
6.) It basically runs this way - when it detects face it means mask is not found, whereas mask on face gets detected. This is highly useful at gate entry allowance but need a better camera and abetter model to best accuracy. However, this model only provides minimal accuracy.  

![Mask Not Detected](https://github.com/deeppatel23/opencvproject/blob/master/masknotdetected.jpeg)
![Mak Detected](https://github.com/deeppatel23/opencvproject/blob/master/maskdetection.py)
