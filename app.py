import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import model_from_json
from pyfirmata import Arduino, OUTPUT

json_file = open("gesture_model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("gesture_model.h5")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_drawing = mp.solutions.drawing_utils

def preprocess_landmarks(landmarks):
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    landmarks = (landmarks - np.mean(landmarks)) / np.std(landmarks)
    return landmarks[:63].reshape(1, -1)

board = Arduino('COM5')
led_pin = 13
board.digital[led_pin].mode = OUTPUT

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        continue
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            preprocessed_landmarks = preprocess_landmarks(hand_landmarks.landmark)
            prediction = model.predict(preprocessed_landmarks)
            predicted_label = np.argmax(prediction)
            
            if predicted_label == 0:
                gesture = "fist"
                board.digital[led_pin].write(0)
            else:
                gesture = "palm"
                board.digital[led_pin].write(1)
                
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (23, 25, 0), 2)
    
    cv2.imshow('Hand Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
board.exit()
