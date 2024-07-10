########### Gark Kazanjian Deep Computer Vision Golf Swing Analysis ###########

########### Gark Kazanjian Deep Computer Vision Golf Swing Analysis ###########

import cv2 as cv
import numpy as np
import mediapipe as mp

# create tracker object
#tracker = EuclideanDistTracker()

# initializing practice image
cap = cv.VideoCapture('gk.avi')


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


# define desired video capture size
frame_width = 640
frame_height = 480

# object detection from stable camera
object_detector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=200)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
# loop to read and display each frame
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        height, width, _ = frame.shape

        # Resize the frame
        frame = cv.resize(frame, (frame_width, frame_height))

        image = cv.cvtColor(frame,cv.COLOR_RGB2BGR)
        results = holistic.process(image)
        
        image = cv.cvtColor(image,cv.COLOR_RGB2BGR)

        # draw the face landmarks
      #  mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        
        # draw the body landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        
        # draw the left and right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv.imshow('image',image)
     #   cv.imshow('frame',frame)

        # extract region of interest
        roi = frame[250:720,200:900] 

        # object detection
        mask = object_detector.apply(frame)
        _, mask = cv.threshold(mask, 254, 255, cv.THRESH_BINARY)
        countours, _  = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for cnt in countours:
            # calculate the area and remove all small elements
            area = cv.contourArea(cnt)
            if area > 100:
                cv.drawContours(frame, [cnt], -1, (0,255,0), 2)
                x, y, w, h = cv.boundingRect(cnt)
                cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
                print(x, y, w, h)

            cv.drawContours(frame, [cnt], -1, (0,255,0), 2)

      #  cv.imshow('Image', image)
        cv.imshow('Frame', frame)
       # cv.imshow('Masked Cap', mask)
      #  cv.imshow('ROI', roi)
        cv.imshow("Pose", frame)




        key = cv.waitKey(30)
        if key ==27:
            break

# release the video capture object and close all OpenCV windows
cap.release()
cv.destroyAllWindows()