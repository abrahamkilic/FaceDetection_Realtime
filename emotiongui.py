from tkinter import *
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import cv2
from emotion import analyze_emotion, face_cascade, model, label_dict
from keras.preprocessing import image as keras_image

# Kamerayı başlat
cap = cv2.VideoCapture(0)

# Kameradan alınan görüntüyü güncelleyen fonksiyon
def update_frame():
    ret, frame = cap.read()  # Kameradan bir frame al
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Renk formatını RGB'ye dönüştür
    emotion_frame = analyze_emotion(frame)  # Duygu analizini yap
    frame = Image.fromarray(frame)  # Görüntüyü PIL formatına dönüştür
    frame = ImageTk.PhotoImage(frame)  # PIL formatını Tkinter formatına dönüştür
    video_label.imgtk = frame  # Görüntüyü etikete yerleştir
    video_label.configure(image=frame)  # Etiketi güncelle
    video_label.after(10, update_frame)  # 10ms sonra tekrar çağrılacak


# Butona tıklandığında duygu analizi yapacak fonksiyon
def analyze_emotion_on_button_click():
    ret, frame = cap.read()  # Kameradan bir frame al
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Gri tonlamalı formata dönüştür
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  # Yüzleri tespit et

    emotion_text = giris.get()  # Entry alanından metni al

    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_roi, (48, 48))
        face_img_array = keras_image.img_to_array(face_img)
        face_img_array = np.expand_dims(face_img_array, axis=0)
        face_img_array /= 255.0

        predictions = model.predict(face_img_array)
        emotion_label_index = np.argmax(predictions)
        predicted_emotion = label_dict[emotion_label_index]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f'{predicted_emotion} - {emotion_text}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame = ImageTk.PhotoImage(frame)
    video_label.imgtk = frame
    video_label.configure(image=frame)

# Tkinter penceresini oluştur
root = tk.Tk()
root.title("Yüz Tanıma Sistemi")

main_frame = Frame(root)
main_frame.pack()

webcam_frame = Frame(main_frame)
webcam_frame.pack(side=LEFT)

# Kameradan alınan görüntüyü göstermek için etiket oluştur
video_label = tk.Label(webcam_frame)
video_label.pack()

update_frame()  # Görüntü güncelleme fonksiyonunu çağır

face_recognition_frame = Frame(main_frame)
face_recognition_frame.pack(side=RIGHT)

etiket = Label(face_recognition_frame, text="Yüz Tanıma Sistemi", font="Verdana 20 bold")
etiket.pack()

# Duygu analizi yapacak buton oluştur
add_buton = Button(face_recognition_frame, text="Yüz Ekle", font="Verdana 10 bold", width=10, height=2, command=analyze_emotion_on_button_click)
add_buton.pack()

liste = Listbox(face_recognition_frame)
liste.pack()

giris = Entry(face_recognition_frame, font="Verdana 10 bold", width=10)
giris.pack()

root.mainloop()  # Pencereyi çalıştır

cap.release()  # Kamerayı kapat
cv2.destroyAllWindows()  # Pencereyi kapat