import cv2
import numpy as np
import pyttsx3
import streamlit as st
from tempfile import NamedTemporaryFile

# Load the pre-trained pedestrian classifier
body_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Create a Streamlit web app
st.title('Pedestrian Detection Web App')

# Initialize video variable
video = None

# Upload a video file or use the webcam
video_option = st.radio('Select Video Source:', ('Upload Video', 'Use Webcam'))

if video_option == 'Upload Video':
    video_file = st.file_uploader('Upload a video file', type=['mp4', 'avi'])
    if video_file is not None:
        # Use a temporary file to store the uploaded video
        temp_file = NamedTemporaryFile(delete=False)
        temp_file.write(video_file.read())
        temp_file.close()
        video = cv2.VideoCapture(temp_file.name)
else:
    video = cv2.VideoCapture(0)  # 0 for the default webcam

# Perform pedestrian detection and display results
scaleFactor = 1.1
minNeighbors = 3

# Check if the video source is opened successfully
if video is None or not video.isOpened():
    st.write("Error: Video source is not available.")
else:
    while True:
        ret, frame = video.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bodies = body_classifier.detectMultiScale(gray, scaleFactor, minNeighbors)
        person_count = len(bodies)

        for (x, y, w, h) in bodies:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        cv2.putText(frame, f'Person Count: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        st.image(frame, channels="BGR", use_column_width=True)

        # Check if the number of pedestrians is more than 3
        if person_count > 3:
            alert_text = "ALERT: More than 3 pedestrians detected!"
            engine.say(alert_text)
            engine.runAndWait()

# Release the video capture and close all windows
if video is not None:
    video.release()
cv2.destroyAllWindows()
