import cv2
import numpy as np
from matplotlib import pyplot as plt

df = cv2.CascadeClassifier("modelo/haarcascade_frontalface_default.xml")

video_lido = cv2.VideoCapture("video.mp4")

template = cv2.imread('imagem/rosto_teste.png',0)
w, h = template.shape[::-1]
# Contador para controlar nome das imagens salvas
contador = 0
while True:
    (sucesso, frame) = video_lido.read() 
    if not sucesso:
        break

    #converte para tons de cinza
    frame_pb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      
    #Histograma Frame


    img = frame_pb
   
    methods = ['cv2.TM_CCOEFF_NORMED']

    for meth in methods:
       
        method = eval(meth)
    # Aplicando o template Matching
        res = cv2.matchTemplate(img,template,method)


    #Recupera a similaridade entre o template e o conteúdo da Imagem de busca
        min_val, similaridade, min_loc, max_loc = cv2.minMaxLoc(res)
        texto = 'Similaridade com {0} entre Imagens é {1}%'.format(meth,round(similaridade*100,2))

        if similaridade > 0.10:
            plt.subplot(121), plt.imshow(template, cmap='gray')
            plt.title('Template'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(img, cmap='gray')
            plt.title('Imagem de Busca'), plt.xticks([]), plt.yticks([])
            plt.suptitle(meth)
            plt.show()
            
        
        
    contador += 1
    #Exibe o frame
    

    #Espera que a tecla 's' seja pressionada para sair
    if cv2.waitKey(1) & 0xFF == ord("s"):
        break


#fecha streaming
video_lido.release()
cv2.destroyAllWindows()
