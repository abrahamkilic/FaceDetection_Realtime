import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk
import cv2
import numpy as np
import face_recognition
import os
import imutils
import time
from imutils.video import VideoStream
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"TensorFlow is using GPU: {physical_devices[0]}")
else:
    print("No GPU devices found. TensorFlow will run on CPU.")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Tkinter penceresini oluştur
root = tk.Tk()
root.title("Yüz Tanıma ve Duygu Analizi")

# Frame'leri oluştur
main_frame = tk.Frame(root)
main_frame.pack()

left_frame = tk.Frame(main_frame)
left_frame.pack(side=tk.LEFT)

right_frame = tk.Frame(main_frame)
right_frame.pack(side=tk.RIGHT)

# Kamera görüntüsü için etiket
video_label = tk.Label(left_frame)
video_label.pack()

# Yüz Tanıma ve Duygu Analizi etiketi
label = tk.Label(right_frame, text="Yüz Tanıma ve Duygu Analizi", font=("Helvetica", 16))
label.pack()

# Uygulamayı kapatacak buton
close_button = tk.Button(right_frame, text="Uygulamayı Kapat", command=root.quit)
close_button.pack()

# Eğitilmiş duygu analizi modelini yükle
model = load_model('model_1.h5')
label_dict = {0: 'Kizgin', 1: 'İgrenme', 2: 'Korku', 3: 'Mutlu', 4: 'Notr', 5: 'Uzgun', 6: 'Saskin'}

# Yüz tanıma için kullanılacak parametreler
DEFAULT_PROTOTXT = "deploy.prototxt.txt"
DEFAULT_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
DEFAULT_CONFIDENCE = 0.5

# Caffe modelini diskten yükle
net = cv2.dnn.readNetFromCaffe(DEFAULT_PROTOTXT, DEFAULT_MODEL)


# Video akışını başlat
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Kamera görüntüsünü güncelle
def update_video():
    frame = vs.read()
    frame = imutils.resize(frame, width=900)
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < DEFAULT_CONFIDENCE:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        face_roi = frame[startY:endY, startX:endX]
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(face_gray, (48, 48))

        face_img_array = image.img_to_array(face_img)
        face_img_array = np.expand_dims(face_img_array, axis=0)
        face_img_array = np.expand_dims(face_img_array, axis=-1)

        predictions = model.predict(face_img_array)
        emotion_label_index = np.argmax(predictions)
        predicted_emotion = label_dict[emotion_label_index]

        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(frame, f'Duygu: {predicted_emotion}', (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Yüz eşleştirme için resimleri yükle
    image_folder = "image"  # Resimlerin bulunduğu klasör

    known_faces = []
    known_face_names = []

    for file in os.listdir(image_folder):
        if file.endswith(".jpg"):
            file_path = os.path.join(image_folder, file)
            img = face_recognition.load_image_file(file_path)
            encoding = face_recognition.face_encodings(img)[0]  # Her bir resmin yüz kodlamasını al
            known_faces.append(encoding)
            known_face_names.append(os.path.splitext(file)[0])  # Dosya adını yüz ismi olarak ekle

    # Yüz eşleştirme
    face_locations = face_recognition.face_locations(frame)
    unknown_face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding in unknown_face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Bilinmiyor"  # Eğer eşleşme yoksa

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Yüzün etrafına isim yazdır
        cv2.putText(frame, name, (startX, startY - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
    video_label.config(image=photo)
    video_label.image = photo
    video_label.after(10, update_video)

# Kamera görüntüsünü göster
update_video()

# Yeni label oluştur ve konumlandır
label_text = tk.Label(right_frame, text="İsim Giriniz")
label_text.pack()

# Yeni metin kutusu oluştur ve konumlandır
textbox = tk.Entry(right_frame)
textbox.pack()
def save_photo():
    # Metin kutusundaki değeri al
    photo_name = textbox.get()

    # Eğer metin kutusu boşsa rastgele bir isimle kaydet
    if not photo_name.strip():
        # Uyarı mesajı
        messagebox.showwarning("Uyarı", "Fotoğraf adı boş olamaz. Lütfen bir ad girin.")
        return

    # Kamera görüntüsünü al
    frame = vs.read()

    # Fotoğrafı kaydet
    image_folder = "C:/Users/Paban/Desktop/FaceDetection_Realtime/image"
    file_path = os.path.join(image_folder, f"{photo_name}.jpg")
    cv2.imwrite(file_path, frame)
def show_photo():
    # Metin kutusundaki değeri al
    photo_name = textbox.get()

    # Eğer metin kutusu boşsa uyarı mesajı göster
    if not photo_name.strip():
        messagebox.showwarning("Uyarı", "Fotoğraf adı boş olamaz. Lütfen bir ad girin.")
        return

    # Dosya yolu oluştur
    image_folder = "C:/Users/Paban/Desktop/FaceDetection_Realtime/image"
    file_path = os.path.join(image_folder, f"{photo_name}.jpg")

    # Fotoğrafın varlığını kontrol et
    if not os.path.isfile(file_path):
        messagebox.showwarning("Uyarı", "Belirtilen isimde bir fotoğraf bulunamadı.")
        return

    # Fotoğrafı göster
    img = Image.open(file_path)
    img.show()

# Yeni butonu oluştur ve fonksiyonu bağla
save_button = tk.Button(right_frame, text="Fotoğrafı Kaydet", command=save_photo)
save_button.pack()

# Yeni butonu oluştur ve fonksiyonu bağla
show_button = tk.Button(right_frame, text="Fotoğrafı Göster", command=show_photo)
show_button.pack()

# Tkinter penceresini çalıştır
root.mainloop()

# Temizlik işlemleri
cv2.destroyAllWindows()
vs.stop()