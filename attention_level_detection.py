# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import dlib
import sys
from imutils import face_utils
from scipy.spatial import distance

VECTOR_SIZE = 3
def queue_in(queue, data):
    ret = None
    if len(queue) >= VECTOR_SIZE:
        ret = queue.pop(0)
    queue.append(data)
    return ret, queue

def eye_aspect_ratio(eye):
    # print(eye)
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

EYE_AR_THRESH = 0.3# EARthreshold
EYE_AR_CONSEC_FRAMES = 3# is EAR threshold is less for 3 frame is blank
EYE_AR_SLEEP_FRAMES = 5# is EAR threshold is less for 8 frame is sleep

# The number of the point sequence
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1

class fatigue(object):

    
    def __init__(self,file_dictory):
#    def __init__(self,file_dictory,landmask_path,facehaar_path,eyehaar_path):
        self.file=os.path.abspath(str(file_dictory))
        os.chdir(self.file)
        self.roi_face=[]
        self.roi_eye=[]
        self.predictor_path=r'/Users/bozhang/Desktop/opencv/555_Final/Bo_Zhang_555_final_project_unchanged/Code/shape_predictor_68_face_landmarks.dat'
#        self.predictor_path=os.path.abspath(str(landmask_path))
        self.face_haar_path=r'/Users/bozhang/Desktop/opencv/555_Final/Bo_Zhang_555_final_project_unchanged/Code/haarcascade_frontalface_default.xml'
#        self.face_haar_path=os.path.abspath(str(facehaar_path))
        self.eye_haar_path=r'/Users/bozhang/Desktop/opencv/555_Final/Bo_Zhang_555_final_project_unchanged/Code/haarcascade_eye.xml'
#        self.eye_haar_path=os.path.abspath(str(eyehaar_path))
    def detect_face(self):
        face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        face.load(self.face_haar_path)
        i=1
        for f in os.listdir(self.file):
            face_image=cv2.imread(f)
            face_image=cv2.medianBlur(face_image,3)
            #change image to gray image
            if face_image.ndim==3:
                face_image_gray=cv2.cvtColor(face_image,cv2.COLOR_BGR2GRAY)
            else:
                face_image_gray=face_image
            faces=face.detectMultiScale(face_image_gray,1.3,5)
            if len(faces)!=0:
                for (x,y,w,h) in faces:
                    self.roi_face.append(face_image[y:y+h,x:x+w,:])
#                    cv2.imwrite(self.file+"\\%g.jpg"%i,face_image_gray[y:y+h,x:x+w])
                    i+=1
        print("the face number %g"%len(self.roi_face))
            #find the image of the
    def detect_eye(self):
        eye=cv2.CascadeClassifier('haarcascade_eye.xml')
        eye.load(self.eye_haar_path)
        i=1
        for face in self.roi_face:
            eyes=eye.detectMultiScale(face,1.03,20,0,(40,40))#(40,40)limited the search area，avoid the noise and mouse to affect the result
            if len(eyes)!=0:
                for (x,y,w,h) in eyes:
                    self.roi_eye.append(face[y:y+h,x:x+w,:])
#                    cv2.imwrite(self.file+"\\%g.jpg"%i,face[y+10:y+h,x+10:x+w,:])
                    i+=1
        print("The number of the eyes in the libary %g"%len(self.roi_eye))
    #the detection of the eye
    def feature_eye(self):
        i=1
        for e in self.roi_eye:
            e_g=cv2.cvtColor(e,cv2.COLOR_BGR2GRAY)
            _,thresh=cv2.threshold(e_g,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            _,cnts,h=cv2.findContours(thresh,0,1)
            cnt_max=sorted(cnts,key=lambda x:cv2.contourArea(x),reverse=True)[0]
            con_hull=cv2.convexHull(cnt_max)
            hull_index=cv2.convexHull(cnt_max,returnPoints = False)
            defects = cv2.convexityDefects(cnt_max,hull_index)
            temp=[]
            point=[]
            for j in range(defects.shape[0]):
                _,_,f,d=defects[j,0]
                point.append(tuple(cnt_max[f][0]))
            for t in point:
                temp.append(sum(t))
            index=np.argsort(temp)
            close=point[index[0]]#Two corner of eye，colse,far
            far=point[index[-1]]
#            np.sort()
            cv2.circle(e,close,5,(0,255,0),-1)
            cv2.circle(e,far,5,(0,255,0),-1)
            cv2.drawContours(e,[con_hull],0,(0,0,255),2)
            cv2.putText(e,str(cv2.contourArea(cnt_max)),(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0))
            cv2.imwrite(self.file+"\\%g.jpg"%i,e)
            i+=1
    def dlib_detect(self):
        frame_counter = 0# counter the frame
        blink_counter = 0# counter the number of the blink
        tired_counter =0;
        tired_frame_counter =0;
        
        detector=dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.predictor_path)
        cap=cv2.VideoCapture(0)#openthe video capture
        if cap.isOpened() is False:
            raise("IO error") #threw the exception
        cap.set(cv2.CAP_PROP_FPS,60)
#        cv2.namedWindow("Capture", cv2.WINDOW_NORMAL)
        forward_left_eye_ratio=None
        forward_right_eye_ratio=None
        flag=0 # in terms of faces number to determine the forward_left....
        while 1:
            ret,frame=cap.read()
            frame=cv2.medianBlur(frame,3)
#            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            if ret==False:
                sys.exit()
            faces=detector(frame,1)#1 is to determine the detectface. 0 is not detect
            if len(faces)>0:
                 if flag==0: #The first frame is the priciple numbers of the faces number.
                     temp=np.zeros((len(faces),1)) #The first use a array to store
                     forward_left_eye_ratio,forward_right_eye_ratio=temp,temp
            else:
#                sys.exit()
#                print("This frame face disapper")
                print("This frame face disapper, show the next frame")
#                break
                continue
            flag=1 #flag=1,if flag = 2, I do not allocate the space，temp
            if len(faces)>0:
                for i,d in enumerate(faces):
                    print('-'*20)
                    # The number of the points numbers
                    shape = predictor(frame, d)
                    # convert the facial landmark (x, y)-coordinates to a NumPy array
                    points = face_utils.shape_to_np(shape)
                    # The left eye point detection
                    leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]
                    # The right eye point detection
                    rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]
                    # calculate the left eye EAR
                    leftEAR = eye_aspect_ratio(leftEye)
                    # calculate the right eye EAR
                    rightEAR = eye_aspect_ratio(rightEye)
                    print('leftEAR = {0}'.format(leftEAR))
                    print('rightEAR = {0}'.format(rightEAR))
                    # The average of the two eye EAR
                    ear = (leftEAR + rightEAR) / 2.0
                    
                    # find the left eye hull
                    leftEyeHull = cv2.convexHull(leftEye)
                    # find the right eye hull
                    rightEyeHull = cv2.convexHull(rightEye)
                    # draw the counter of the left eye
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    # draw the counter of the right eye
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
                    # If EAR is less than the threshold, calculate the numbers of the sequence of the frame, if the total number is larger than EYE_AR_CONSEC_FRAMES, it should be a blink
                    if ear < EYE_AR_THRESH:
                        frame_counter += 1
                    else:
                        if frame_counter >= EYE_AR_CONSEC_FRAMES:
                            blink_counter += 1
                        frame_counter = 0
                    
                    if ear < EYE_AR_THRESH:
                        tired_frame_counter +=1;
                    else:
                        if tired_frame_counter >= EYE_AR_SLEEP_FRAMES:
                            tired_counter +=1;
                        tired_frame_counter = 0;
        
                    # show the result of the blink_counte and EAR
                    cv2.putText(frame, "Blinks:{0}".format(blink_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.putText(frame, "EAR:{:.2f}".format(ear), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.putText(frame, "Tired time:{0}".format(tired_counter), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                    cv2.putText(frame, "EAR_tired:{:.2f}".format(ear), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                    cv2.putText(frame, "EAR_tired frame:{:.2f}".format(tired_frame_counter), (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                    #draw the face are of the detected face
                    cv2.rectangle(frame,(d.left(),d.top()),(d.right(),d.bottom()),(0,255,0),2)
                    cv2.putText(frame,str(i+1),(d.left()-10,d.top()-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                    #find the face recognise point
                    shape=predictor(frame,d)
#                    print(shape,type(shape))
                    #cover the face point in to the （x,y)
                    points = face_utils.shape_to_np(shape)
                    long_left=self.distance(points[36,:],points[39,:])
                    short_left=self.distance((points[37,:]+points[38,:])/2,(points[41,:]+points[40,:])/2)
                    long_right=self.distance(points[42,:],points[45,:])
                    short_right=self.distance((points[43,:]+points[44,:])/2,(points[46,:]+points[47,:])/2)
                    
                    #find the eye height and width comparation, than find the next frame. after continue stop the loop
                    if forward_left_eye_ratio[i]==0 and forward_right_eye_ratio[i]==0:
                        forward_left_eye_ratio[i]=short_left/long_left
                        forward_right_eye_ratio[i]=short_right/long_right
                        #find the next faces' eye height and width comparation
                        continue 
                    ##下一帧
                    left_eye_ratio_now=np.zeros((forward_left_eye_ratio.shape))
                    right_eye_ratio_now=np.zeros((forward_right_eye_ratio.shape))
                    left_eye_ratio_now[i]=short_left/long_left
                    right_eye_ratio_now[i]=short_right/long_right
                    print("The unmber %g people's faces' eye height and width comparation:%g"%(i+1,abs(left_eye_ratio_now[i]-forward_left_eye_ratio[i])))
                    if abs(left_eye_ratio_now[i]-forward_left_eye_ratio[i])>0.2:
                        print("The unmber %g people's left eye changed %g"%(i+1,abs(left_eye_ratio_now-forward_left_eye_ratio)))
                    if abs(right_eye_ratio_now[i]-forward_right_eye_ratio[i])>0.2:
                        print("The unmber %g people's right eye changed %g"%(i+1,abs(right_eye_ratio_now-forward_right_eye_ratio)))
                    if abs(left_eye_ratio_now[i]-forward_left_eye_ratio[i])>0.2 and abs(right_eye_ratio_now[i]-forward_right_eye_ratio[i])>0.2:
                        print("The unmber %g people are tired, you have to have a sleep. "%(i+1))
                        cv2.putText(frame, "The unmber %g people are tired, you have to have a sleep. "%(i+1), (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                    forward_left_eye_ratio[i]=left_eye_ratio_now[i]
                    forward_right_eye_ratio[i]=right_eye_ratio_now[i]
            cv2.imshow("Capture",frame)
            k=cv2.waitKey(10)
            if k==27:
                break
        cap.release()
        cv2.destroyAllWindows()
    def distance(self,p1,p2):
        return np.sqrt(np.sum((p1-p2)*(p1-p2)))


if __name__=="__main__":
#    param=sys.argv[1]
#    print("cmd---Format--python '1.****.py' '2.The location of the face library' '3.The location of the 68_face_landmarks.dat' '4.The location of haarcascade_frontalface_default.xml'  '5.The location of haarcascade_eye.xml'")
    print("md---Format--python '1.****.py' '2.The location of the face library\n' ");
    if len(sys.argv)!=2:
        print("param not enough \n")
#    fold_param=r'Users/bozhang/Desktop/eye_detection/my_photo'
#    fatigue_drive=fatigue(fold_param)
    fatigue_drive=fatigue(sys.argv[1])
    print("**********Eye detection of the corner of the eye*************\n")
    fatigue_drive.detect_face()
    fatigue_drive.detect_eye()
    fatigue_drive.feature_eye()
    print("*************   Attandance level detection     ************\n")
    fatigue_drive.dlib_detect()



