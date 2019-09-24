import cv2
import numpy as np
from matplotlib import pyplot as plt

df = cv2.CascadeClassifier("modelo/haarcascade_frontalface_default.xml")

video_lido = cv2.VideoCapture("video.mp4")


# Contador para controlar nome das imagens salvas
contador = 0
while True:
    (sucesso, frame) = video_lido.read() 
    if not sucesso:
        break

    #converte para tons de cinza
    frame_pb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #detecta os rostos no frame
    faces = df.detectMultiScale(frame_pb, scaleFactor = 1.1, minNeighbors=3, minSize=(5,5), flags=cv2.CASCADE_SCALE_IMAGE)
    frame_temp = frame.copy()
    
    #Histograma Frame
    histg = cv2.calcHist([frame_temp],[0],None,[256],[0,256])
    plt.plot(histg)
    plt.savefig('Hist/histograma' + str(contador) + '.png')
    plt.clf()
    # Cria os retângulos
    for (x, y, w, h) in faces:
        img = cv2.rectangle(frame_temp, (x, y), (x + w, y + h), (0, 255, 255), 2) # RGB
        # Salva cada frame com rosto
        roi_color = img[y:y+h, x:x+w]
    
        cv2.imwrite("Pessoas/rosto"+str(contador)+".png", roi_color) 
        
        histFace = cv2.calcHist([roi_color],[0],None,[256],[0,256])
        plt.plot(histFace)
        plt.savefig('HistFace/histograma' + str(contador) + '.png')
        plt.clf()
        # FIM HISTOGRAMA
        
    contador += 1
    #Exibe o frame
    cv2.imshow("aula_face...", frame_temp)

    #Espera que a tecla 's' seja pressionada para sair
    if cv2.waitKey(1) & 0xFF == ord("s"):
        break


#fecha streaming
video_lido.release()
cv2.destroyAllWindows()