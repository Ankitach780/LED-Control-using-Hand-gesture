import cv2
import mediapipe as mp
import numpy as np
import os
import joblib
import pandas as pd
import os

dir = 'HandGesture'
filepaths = []
labels = []

folds = os.listdir(dir)
for fold in folds:
    foldpath = os.path.join(dir, fold)
    filelist = os.listdir(foldpath)
    for file in filelist:
        fpath = os.path.join(foldpath, file)
        filepaths.append(fpath)
        labels.append(fold)

Fseries = pd.Series(filepaths, name= 'Gesture')
Lseries = pd.Series(labels, name='Labels')
df = pd.concat([Fseries, Lseries], axis= 1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
data = []

def preprocess_landmarks(landmarks):
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    landmarks = (landmarks - np.mean(landmarks)) / np.std(landmarks)
    return landmarks

labels_dict = {'no': 0, 'yes': 1}

for index, row in df.iterrows():
    image_path = row['Gesture']
    label = row['Labels']
    
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to load image {image_path}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            preprocessed_landmarks = preprocess_landmarks(hand_landmarks.landmark)
            data.append((preprocessed_landmarks, labels_dict[label]))
print(data)

joblib.dump(data, 'hand_gesture_data.pkl')

print("Data processing complete.")
