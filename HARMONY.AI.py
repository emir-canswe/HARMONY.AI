import cv2 as cv

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)  # Görüntüyü ters çevir (ayna gibi)

    cv.imshow("Aynalanmış Kamera", frame)

    if cv.waitKey(1) & 0xFF == 27:  # ESC tuşu ile çık
        break

cap.release()
cv.destroyAllWindows()
