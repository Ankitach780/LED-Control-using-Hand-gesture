import joblib

data = joblib.load('hand_gesture_data.pkl')

for landmarks, label in data[:5]:
    print(f"Landmarks: {landmarks}")
    print(f"Label: {label}")
