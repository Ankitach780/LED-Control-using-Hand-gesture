import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import model_from_json
from pyfirmata import Arduino, OUTPUT

# Load the trained gesture model
json_file = open("gesture_model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("gesture_model.h5")

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Preprocessing function for landmarks
def preprocess_landmarks(landmarks):
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    landmarks = (landmarks - np.mean(landmarks)) / np.std(landmarks)
    return landmarks[:63].reshape(1, -1)

# Setup Arduino
board = Arduino('COM3')
led_pin = 13
board.digital[led_pin].mode = OUTPUT

# Variable to store LED state
led_state = 0  # 0: Off, 1: On

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        continue
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Preprocess landmarks and predict gesture
            preprocessed_landmarks = preprocess_landmarks(hand_landmarks.landmark)
            prediction = model.predict(preprocessed_landmarks)
            predicted_label = np.argmax(prediction)
            
            if predicted_label == 0:  # Fist gesture
                gesture = "fist"
                led_state = 1  # Turn on LED
            elif predicted_label == 1:  # Palm gesture
                gesture = "palm"
                led_state = 0  # Turn off LED
            
            # Set LED state on Arduino
            board.digital[led_pin].write(led_state)
            
            # Display gesture on frame
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (23, 25, 0), 2)
    
    # Display the frame
    cv2.imshow('Hand Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q'
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
board.exit()
