import cv2 as cv#open cv kur

cap = cv.VideoCapture(0)#kameraya baglan

while True:#esc tusuna basaa-na kadar sonsuz dongu olustır
    ret, frame = cap.read()#kerayi oku
    if not ret:
        break

    frame = cv.flip(frame, 1)  # Görüntüyü ters çevir (ayna gibi)

    cv.imshow("Aynalanmış Kamera", frame)#kemrayi goster

    if cv.waitKey(1) & 0xFF == 27:  # ESC tuşu ile çık
        break

cap.release()#kemraya ulasmanini engelle
cv.destroyAllWindows()
