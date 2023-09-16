
import cv2
import pyttsx3

# Load the pre-trained pedestrian classifier
body_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

def detect_pedestrians(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_classifier.detectMultiScale(gray, scaleFactor, minNeighbors)

    person_count = len(bodies)  # Count the number of detected pedestrians

    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(frame, f'Person Count: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame, person_count

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Open the video capture
video = cv2.VideoCapture('D:\DLprojects\pedestrian_detection\walking.avi')

while True:  # Use an infinite loop for video processing
    ret, frame = video.read()

    if not ret:
        break

    scaleFactor = 1.1
    minNeighbors = 3

    result_frame, person_count = detect_pedestrians(frame)

    # Check if the number of pedestrians is more than 3
    if person_count > 3:
        alert_text = "ALERT: More than 3 pedestrians detected!"
        alert_font = cv2.FONT_HERSHEY_SIMPLEX
        alert_font_scale = 1
        alert_font_color = (0, 0, 255)
        alert_thickness = 2
        text_size = cv2.getTextSize(alert_text, alert_font, alert_font_scale, alert_thickness)[0]
        text_x = (result_frame.shape[1] - text_size[0]) // 2
        text_y = (result_frame.shape[0] + text_size[1]) // 2
        #draw text on frame:
        cv2.putText(result_frame, alert_text, (text_x, text_y), alert_font, alert_font_scale, alert_font_color, alert_thickness)
        
        engine.say(alert_text)
        engine.runAndWait()

    cv2.imshow('Pedestrians', result_frame)

    # Check for the 'q' key press and exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video.release()
cv2.destroyAllWindows()
