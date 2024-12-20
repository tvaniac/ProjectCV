import streamlit as st
import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
from threading import Thread
import time
import winsound

# Constants
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20 

# Global variables
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0

# Functions
def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    while alarm_status:
        winsound.Beep(1000, 500)

    if alarm_status2:
        saying = True
        winsound.Beep(1500, 500)
        saying = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance

# Load detector and predictor
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Streamlit UI
st.title("Drowsiness and Yawn Detection")
st.markdown("**Check the box below to start the camera:**")

FRAME_WINDOW = st.image([])
run = st.checkbox("Run camera", key="run_camera")

# Video capture
if "cap" not in st.session_state:
    st.session_state.cap = None

if run:
    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)
        st.success("Camera started!")

    while run:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("Failed to open webcam.")
            break

        frame = cv2.resize(frame, (450, 300))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape) 

            ear, leftEye, rightEye = final_ear(shape)
            distance = lip_distance(shape)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not alarm_status:
                        alarm_status = True
                        Thread(target=alarm, args=('Drowsiness Alert',)).start()

                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                alarm_status = False

            if distance > YAWN_THRESH:
                cv2.putText(frame, "Yawn Alert", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not alarm_status2 and not saying:
                    alarm_status2 = True
                    Thread(target=alarm, args=('Yawn Alert',)).start()
            else:
                alarm_status2 = False

            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"YAWN: {distance:.2f}", (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    st.session_state.cap.release()
    st.session_state.cap = None
    FRAME_WINDOW.image([])
    st.checkbox("Run camera", value=False, key="run_camera")  # Uncheck the checkbox automatically
else:
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
        FRAME_WINDOW.image([])
    st.info("Check 'Run camera' to start detection.")
