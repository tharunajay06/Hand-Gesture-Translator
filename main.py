import cv2
import mediapipe as mp
import numpy as np
import time
import os
from gtts import gTTS
import joblib

# Load model and label encoder
model = joblib.load("models/gesture_model.pkl")
encoder = joblib.load("models/gesture_label_encoder.pkl")

# Load labels in original order
labels = encoder.classes_
print("Labels:", labels)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Sentence construction
collected_gestures = []
prev_prediction = ""
start_time = time.time()
max_gestures = 3
max_time = 20  # seconds

# Webcam
cap = cv2.VideoCapture(0)

def speak(sentence):
    tts = gTTS(text=sentence, lang='en')
    tts.save("sentence.mp3")
    os.system("ffplay -nodisp -autoexit sentence.mp3")

def draw_progress_bar(image, start_time, max_time):
    elapsed = time.time() - start_time
    progress = min(elapsed / max_time, 1.0)
    bar_width = int(progress * 300)
    cv2.rectangle(image, (10, 450), (10 + bar_width, 470), (0, 255, 0), -1)
    cv2.rectangle(image, (10, 450), (310, 470), (255, 255, 255), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 42 keypoints
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            if len(landmarks) == 42:
                prediction = model.predict([landmarks])[0]
                gesture_label = encoder.inverse_transform([prediction])[0]

                # Show prediction on screen
                cv2.putText(image, f"{gesture_label}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                if gesture_label != prev_prediction:
                    prev_prediction = gesture_label

                    if gesture_label == "clear":
                        print("🧹 CLEAR gesture detected. Resetting.")
                        collected_gestures = []
                        start_time = time.time()
                        continue

                    collected_gestures.append(gesture_label)
                    print("✅ Collected gesture:", gesture_label)

    # Show current sentence
    cv2.putText(image, "Sentence: " + " ".join(collected_gestures), (10, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show time progress
    draw_progress_bar(image, start_time, max_time)

    cv2.imshow("Real-time Gesture Translator", image)

    # Time/gesture limit reached
    if len(collected_gestures) >= max_gestures or (time.time() - start_time) > max_time:
        if collected_gestures:
            sentence = " ".join(collected_gestures)
            print("🗣 Speaking:", sentence)
            speak(sentence)
        # Reset
        collected_gestures = []
        prev_prediction = ""
        start_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
