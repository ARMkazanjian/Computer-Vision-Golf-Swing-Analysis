########### Gark Kazanjian Computer Vision Golf Swing Analysis ###########

import cv2 as cv
import numpy as np
import mediapipe as mp

# initializing practice image
cap = cv.VideoCapture('gk_hit.avi')
print(cap)

# angle calculator function for body edges
def calculate_angle(a, b, c):
    a = np.array(a) # First point
    b = np.array(b) # Mid point
    c = np.array(c) # End point

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# used to draw the connections obtained from various MediaPipe solutions onto the images
mp_drawing = mp.solutions.drawing_utils 
# simultaneously detects body, face, hands, and pose landmarks in an image
mp_holistic = mp.solutions.holistic


# define desired video capture size
frame_width = 400
frame_height = 550

# object detection from stable camera
object_detector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=300)

# body pose detection confidence model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
# loop to read and display each frame
    while True:
      
        # capture frame-by-frame
        ret, frame = cap.read()


        image = cv.cvtColor(frame,cv.COLOR_RGB2BGR)
        results = holistic.process(image)
        
        image = cv.cvtColor(image,cv.COLOR_RGB2BGR)

        # Resize the frame
        image = cv.resize(image, (frame_width, frame_height))

        # draw the body landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        
        # draw the left and right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv.imshow('image',image)

        ################################################## calculate angles between shoulder, elbow and wrist
        try:
            landmarks = results.pose_landmarks.landmark

                    # Left side landmarks
            left_hip = [landmarks[mp_holistic.PoseLandmark.LEFT_HIP].x, landmarks[mp_holistic.PoseLandmark.LEFT_HIP].y]
            left_knee = [landmarks[mp_holistic.PoseLandmark.LEFT_KNEE].x, landmarks[mp_holistic.PoseLandmark.LEFT_KNEE].y]
            left_ankle = [landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_holistic.PoseLandmark.LEFT_KNEE].y]

            #right_hip = landmarks[mp_holistic.PoseLandmark.RIGHT_HIP]
            #r_hip = (right_hip.x, right_hip.y)
            left_hip_angle = calculate_angle(left_ankle, left_hip, left_knee)   # Angle between left knee-hip-right hip

            # print angle for right elbow
            cv.putText(image, str(left_hip_angle), 
                      tuple(np.multiply(left_hip, [frame_width, frame_height]).astype(int)), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)


            # get the coordinates for left side of body
            shoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]

              
            # calculate the angle
            angle_elbow = calculate_angle(shoulder, elbow, wrist)
            #amgle_shoulder = calculate_angle()

            # print angle for elbow
            #cv.putText(image, str(angle_elbow), 
             #          tuple(np.multiply(elbow, [frame_width, frame_height]).astype(int)), 
             #          cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
            
            # ... existing code ...

            # get the coordinates for right side of body
            shoulder_right = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_right = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_right = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y]

            # calculate the angle
            angle_elbow_right = calculate_angle(shoulder_right, elbow_right, wrist_right)

            # print angle for right elbow
            cv.putText(image, str(angle_elbow_right), 
                      tuple(np.multiply(elbow_right, [frame_width, frame_height]).astype(int)), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)

            # ... existing code ...

        except:
            pass
        
        ################################################## end 

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

        # ... existing code ...

        # get the coordinates for right and left foot index
        right_foot_index = [landmarks[mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
        left_foot_index = [landmarks[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX.value].y]

        # calculate the distance in pixels
        distance_pixels = np.sqrt((right_foot_index[0] - left_foot_index[0])**2 + (right_foot_index[1] - left_foot_index[1])**2)

        # define your scale factor (inches per pixel)
        scale_factor = 100.00  # replace with your actual scale factor

        # calculate the distance in inches
        distance_inches = distance_pixels * scale_factor

        # print distance
        cv.putText(image, str(distance_inches) + " inches", 
                tuple(np.multiply(right_foot_index, [frame_width, frame_height]).astype(int)), 
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)

        # ... existing code ...

        
        # displaying newly rendered videos
        cv.imshow('Image', image)
        #cv.imshow('Frame', frame)
       # cv.imshow('Masked Cap', mask)
        #cv.imshow('ROI', roi)
      #  cv.imshow("Pose", frame)

        key = cv.waitKey(30)
        if key ==27:
            break

# release the video capture object and close all OpenCV windows
cap.release()
cv.destroyAllWindows()