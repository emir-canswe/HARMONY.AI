"""
isaret_dil_tanima.py
Tek dosyada tam proje: veri toplama + hazırlık + model eğitimi + gerçek zamanlı tanıma
Gereken kütüphaneler:
pip install opencv-python mediapipe numpy pandas tensorflow scikit-learn gTTS playsound
"""

import cv2
import mediapipe as mp
import numpy as np
import os, glob, time, tempfile, argparse
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from gtts import gTTS
try:
    from playsound import playsound
except:
    playsound = None

mp_holistic = mp.solutions.holistic

# ---------- Ortak yardımcı fonksiyonlar ----------
def extract_landmarks(results):
    """Pose + ellerden özellik vektörü oluşturur"""
    v = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            v += [lm.x, lm.y, lm.z, lm.visibility]
    else:
        v += [0]*33*4
    for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if hand:
            for lm in hand.landmark:
                v += [lm.x, lm.y, lm.z]
        else:
            v += [0]*21*3
    return np.array(v, dtype=np.float32)

def speak(text):
    """Metni sesli oku (gTTS ile)"""
    try:
        tts = gTTS(text=text, lang='tr')
        fd, path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        tts.save(path)
        if playsound:
            playsound(path)
        else:
            os.system(f"start {path}" if os.name == 'nt' else f"mpg123 {path}")
        os.remove(path)
    except Exception as e:
        print("Ses çalma hatası:", e)

# ---------- 1. Veri Toplama ----------
def collect_data(label, count=100, seq_len=30, out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        seq = deque(maxlen=seq_len)
        saved = 0
        print("Kamera açıldı, veri toplanıyor... ESC ile çık.")
        while cap.isOpened() and saved < count:
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False
            results = holistic.process(img)
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(img, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(img, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

            vect = extract_landmarks(results)
            seq.append(vect)

            cv2.putText(img, f"Label: {label}  ({saved}/{count})", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow("Veri Toplama", img)

            if cv2.waitKey(1) & 0xFF == 27:
                break
            if len(seq) == seq_len:
                arr = np.array(seq)
                np.save(os.path.join(out_dir, f"{label}_{saved:04d}.npy"), arr)
                saved += 1
                seq.clear()
                time.sleep(0.4)
        cap.release()
        cv2.destroyAllWindows()
        print("Veri toplama tamamlandı.")

# ---------- 2. Veri Hazırlama ----------
def prepare_dataset(data_dir="data", seq_len=30, test_size=0.2, out_dir="prepared"):
    os.makedirs(out_dir, exist_ok=True)
    X, y = [], []
    files = glob.glob(os.path.join(data_dir, "*.npy"))
    for f in files:
        arr = np.load(f)
        label = os.path.basename(f).split("_")[0]
        X.append(arr)
        y.append(label)
    X, y = np.array(X), np.array(y)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=test_size, stratify=y_enc, random_state=42)
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)
    pd.DataFrame({"label": le.classes_, "enc": range(len(le.classes_))}).to_csv(os.path.join(out_dir, "labels.csv"), index=False)
    print("Veri hazırlandı:", X.shape, "etiketler:", le.classes_)

# ---------- 3. Model Eğitimi ----------
def build_model(seq_len, feat_dim, n_classes):
    inp = layers.Input(shape=(seq_len, feat_dim))
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def train_model(prepared_dir="prepared", epochs=40, batch=16, model_out="model_best.h5"):
    X_train = np.load(os.path.join(prepared_dir, "X_train.npy"))
    X_test = np.load(os.path.join(prepared_dir, "X_test.npy"))
    y_train = np.load(os.path.join(prepared_dir, "y_train.npy"))
    y_test = np.load(os.path.join(prepared_dir, "y_test.npy"))
    labels = pd.read_csv(os.path.join(prepared_dir, "labels.csv"))["label"].values
    model = build_model(X_train.shape[1], X_train.shape[2], len(labels))
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=epochs, batch_size=batch)
    model.save(model_out)
    print("Model eğitildi ve kaydedildi:", model_out)

# ---------- 4. Gerçek Zamanlı Tanıma ----------
def realtime_recognizer(model_path="model_best.h5", labels_csv="prepared/labels.csv", seq_len=30, threshold=0.7):
    labels = list(pd.read_csv(labels_csv)["label"])
    model = load_model(model_path)
    cap = cv2.VideoCapture(0)
    seq = deque(maxlen=seq_len)
    last_pred, last_time = None, 0
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False
            results = holistic.process(img)
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(img, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(img, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

            vect = extract_landmarks(results)
            seq.append(vect)

            if len(seq) == seq_len:
                X = np.expand_dims(np.array(seq), axis=0)
                probs = model.predict(X, verbose=0)[0]
                idx = np.argmax(probs)
                prob = probs[idx]
                label = labels[idx]
                if prob > threshold:
                    cv2.putText(img, f"{label} ({prob:.2f})", (10,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    now = time.time()
                    if label != last_pred or now - last_time > 2.0:
                        print("Tahmin:", label, prob)
                        speak(label)
                        last_pred, last_time = label, now
                else:
                    cv2.putText(img, "Tanımsız", (10,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Gerçek Zamanlı Tanıma", img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()

# ---------- Ana Menü ----------
if __name__ == "__main__":
    print("\n=== TÜRK İŞARET DİLİ TANIMA SİSTEMİ ===")
    print("1 - Veri Toplama")
    print("2 - Veriyi Hazırla")
    print("3 - Model Eğit")
    print("4 - Gerçek Zamanlı Tanıma")
    secim = input("Seçim yap (1-4): ").strip()

    if secim == "1":
        lbl = input("Etiket (örnek: Merhaba): ")
        collect_data(lbl)
    elif secim == "2":
        prepare_dataset()
    elif secim == "3":
        train_model()
    elif secim == "4":
        realtime_recognizer()
    else:
        print("Geçersiz seçim.")
