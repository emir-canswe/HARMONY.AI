import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from gtts import gTTS
from playsound import playsound
import os
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ✅ MODEL YÜKLEME
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6
)
hands = vision.HandLandmarker.create_from_options(options)

# ✅ Veri listeleri
sample_data = []
sample_labels = []
model = KNeighborsClassifier(n_neighbors=3)

def add_sample(landmarks, label):
    coords = np.array([[lm.x, lm.y] for lm in landmarks]).flatten()
    sample_data.append(coords)
    sample_labels.append(label)

def predict(landmarks):
    coords = np.array([[lm.x, lm.y] for lm in landmarks]).flatten().reshape(1, -1)
    return model.predict(coords)[0]

def speak(text):
    tts = gTTS(text=text, lang='en')
    filename = "temp.mp3"
    tts.save(filename)
    playsound(filename)
    os.remove(filename)

# ✅ Kamera başlat
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Kamera açılamadı!")
    exit()

recording = False
trained = False
gesture_key = None
last_prediction = ""
last_time = time.time()

print("[h] Hello")
print("[y] Help")
print("[w] Water")
print("[r] Kayıt Aç/Kapat")
print("[s] Model Eğit")
print("[q] Çıkış")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = hands.detect(mp_image)

    if results.hand_landmarks:
        for hand in results.hand_landmarks:

            for lm in hand:
                cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)

            if recording and gesture_key:
                add_sample(hand, gesture_key)
                cv2.putText(frame, "Kayıt: " + gesture_key, (10,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            if trained:
                pred = predict(hand)
                cv2.putText(frame, "Tahmin: " + pred, (10,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

                if pred != last_prediction or time.time() - last_time > 3:
                    speak(pred)
                    last_prediction = pred
                    last_time = time.time()

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'): break
    if key == ord('r'): recording = not recording
    if key in [ord('h'), ord('y'), ord('w')]:
        gesture_key = {'h': 'hello','y':'help','w':'water'}[chr(key)]
    if key == ord('s') and len(sample_data) > 0:
        model.fit(sample_data, sample_labels)
        trained = True
        print("✅ Model Eğitildi!")

    cv2.imshow("Harmony.AI", frame)

cap.release()
cv2.destroyAllWindows()
print("✅ Program sonlandırıldı")
print("emirooooadknadjvbdsjhbvo")
