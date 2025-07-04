import cv2

kamera = cv2.VideoCapture(0)

yuzAlgilayici = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    basariliMi, kare = kamera.read()
    if not basariliMi:
        break

    griKare = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
    yuzler = yuzAlgilayici.detectMultiScale(griKare, 1.3, 5)

    for (x, y, genislik, yukseklik) in yuzler:
        yuzBolgesi = kare[y:y + yukseklik, x:x + genislik]
        bulanikYuz = cv2.GaussianBlur(yuzBolgesi, (75, 75), 30)
        kare[y:y + yukseklik, x:x + genislik] = bulanikYuz

    cv2.imshow("Yuz Bulaniklastirma", kare)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()
