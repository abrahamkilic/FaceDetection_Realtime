import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Eğitilmiş modeli yükle
model = load_model('model_1.h5')



# Etiket sözlüğünü tanımla
label_dict = {0: 'Kizgin', 1: 'İgrenme', 2: 'Korku', 3: 'Mutlu', 4: 'Notr', 5: 'Uzgun', 6: 'Saskin'}

# Webcam'e bağlantı aç (0 varsayılan kamerayı temsil eder)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Yüz tespiti için çerçeveyi gri tonlamaya çevir
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Çerçevedeki yüzleri tespit et
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Çerçeveden yüz bölgesini çıkart
        face_roi = gray_frame[y:y+h, x:x+w]

        # Tahmin için yüzü işle
        face_img = cv2.resize(face_roi, (48, 48))
        face_img_array = image.img_to_array(face_img)
        face_img_array = np.expand_dims(face_img_array, axis=0)
        face_img_array /= 255.0

        # Tahmin yap
        predictions = model.predict(face_img_array)
        emotion_label_index = np.argmax(predictions)

        # İndex'i karşılık gelen duygu etiketine eşleştir
        predicted_emotion = label_dict[emotion_label_index]

        # Tespit edilen yüzün etrafına bir dikdörtgen çiz
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Tahmin edilen duyguyu yüzün yakınına yazdır
        cv2.putText(frame, f'Duygu: {predicted_emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Çerçeveyi görüntüle
    cv2.imshow('Webcam Duygu Analizi', frame)

    # 'q' tuşuna basılırsa döngüyü kır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Webcam'i serbest bırak ve tüm OpenCV pencerelerini kapat
cap.release()
cv2.destroyAllWindows()
