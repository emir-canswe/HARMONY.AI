import cv2  # Kamera ve görüntü işleme kütüphanesi (OpenCV)
import mediapipe as mp  # El takibi ve işaret noktası tespiti için Google kütüphanesi
import numpy as np  # Sayısal işlemler ve diziler için
from sklearn.neighbors import KNeighborsClassifier  # Hareketleri sınıflandırmak için ML algoritması
from gtts import gTTS  # Yazıyı sese çeviren Google Text-to-Speech kütüphanesi
from playsound import playsound  # MP3 dosyasını oynatmak için
import os  # Dosya işlemleri için
import time  # Zaman kontrolü için

# El tanıma sistemi başlatılıyor
mp_hands = mp.solutions.hands  # El takip modülü çağrılır
hands = mp_hands.Hands(        # El takip yapılandırması
    static_image_mode=False,   # Gerçek zamanlı video akışı (True olursa sadece foto)
    max_num_hands=1,           # En fazla 1 el algılansın
    min_detection_confidence=0.7,  # El tespiti için minimum güven
    min_tracking_confidence=0.7   # Elin hareketini takip etmek için minimum güven
)
mp_draw = mp.solutions.drawing_utils  # Ekrana el iskeletini çizdirmek için

# Eğitim verisi ve etiketlerini tutan listeler
sample_data = []  # El pozisyon verileri
sample_labels = []  # El hareketlerinin etiketleri (hello, help, water gibi

# Kayıt modunda her bir pozisyonu listeye ekleyen fonksiyon
def add_sample(landmarks, label):  # landmarks: el noktaları, label: hareket etiketi
    flat = np.array([[lm.x, lm.y] for lm in landmarks]).flatten()  # El noktalarını (x, y) olarak düzleştir
    sample_data.append(flat)  # Veriyi ekle
    sample_labels.append(label)  # Etiketini ekle

# KNN modelini tanımla (3 komşulu)
model = KNeighborsClassifier(n_neighbors=3)

# Eğitilen modele göre tahmin yapan fonksiyon
def predict(landmarks):
    flat = np.array([[lm.x, lm.y] for lm in landmarks]).flatten().reshape(1, -1)  # Giriş verisini düzleştir
    return model.predict(flat)[0]  # Tahmini döndür (örn. 'hello')

# Tahmini sesli söyleyen fonksiyon
def speak(text):
    tts = gTTS(text=text, lang='en')  # Yazıyı İngilizce sese çevir
    filename = "temp.mp3"  # Geçici dosya adı
    tts.save(filename)  # Dosyayı kaydet
    playsound(filename)  # Sesi çal
    os.remove(filename)  # Dosyayı sil

# Kamerayı başlat
cap = cv2.VideoCapture(0)  # 0: varsayılan kamera
recording = False  # Başta kayıt modu kapalı
trained = False  # Başta model eğitilmedi
gesture_key = None  # Hangi hareketin kaydedildiğini tutar
last_prediction = ""  # Son söylenen kelime
last_time_spoken = time.time()  # Son sesli konuşma zamanı

# Bilgilendirme mesajı
print("El hareketlerini kaydetmek için:\n[h] hello\n[y] help\n[w] water\n[s] eğitim\n[q] çıkış\n[r] kayıt açık/kapat")

# Ana döngü (kameradan sürekli görüntü alma)
while True:
    ret, frame = cap.read()  # Kameradan bir kare oku
    if not ret:
        break  # Kamera görüntü vermiyorsa çık

    frame = cv2.flip(frame, 1)  # Aynalı görüntü (kullanıcı gibi görünür)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR → RGB çevir (mediapipe için)
    results = hands.process(rgb)  # El tespiti yap

    # Eğer bir el algılandıysa
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # El iskeletini çiz

            # Kayıt modu açıksa ve bir tuş atanmışsa veri topla
            if recording and gesture_key:
                add_sample(hand_landmarks.landmark, gesture_key)  # Pozisyonu kaydet
                cv2.putText(frame, f"Kaydedildi: {gesture_key}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Ekrana yazı yaz

            # Eğer model eğitildiyse hareket tahmini yap
            if trained:
                prediction = predict(hand_landmarks.landmark)  # Hareketi tahmin et
                cv2.putText(frame, f"Tespit: {prediction}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # Ekrana göster
                
                # Aynı kelime tekrar tekrar söylenmesin diye kontrol
                if prediction != last_prediction or (time.time() - last_time_spoken > 3):
                    speak(prediction)  # Tahmini seslendir
                    last_prediction = prediction  # Son tahmini güncelle
                    last_time_spoken = time.time()  # Konuşma zamanını güncelle

    # Klavyeden tuş kontrolüs
    key = cv2.waitKey(1) & 0xFF  # Tuşa basıldı mı?
    if key == ord('q'):  # Çıkış için
        break
    elif key == ord('r'):  # Kayıt modunu aç/kapat
        recording = not recording
    elif key in [ord('h'), ord('y'), ord('w')]:  # Hangi hareketin kaydedileceğini belirle
        gesture_key = {'h': 'hello', 'y': 'help', 'w': 'water'}[chr(key)]
    elif key == ord('s') and len(sample_data) > 0:  # 's' tuşuna basıldıysa ve veri varsa modeli eğit
        model.fit(sample_data, sample_labels)  # Modeli eğit
        trained = True  # Model eğitildi olarak işaretle
        print("Model eğitildi!")

    # Kameradan alınan görüntüyü göster
    cv2.imshow("HARMONY.AI - Only Hands", frame)

# Döngüden çıkınca her şeyi kapat
cap.release()  # Kamerayı kapat
cv2.destroyAllWindows()  # Açık pencereleri kapat
print("Hello Harmony AI!")
