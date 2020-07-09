import os
import cv2
import numpy as np
import matplotlib.image as img


# La siguiente función de carga de datos se reemplazó por la función de ImageDAtaGenerator que redimensiona las imágenes
def load_data_image(filepath, name, print_dim=False):
    
    """ 
    Argumetos: (Path de la cámara, nombre de persona, imprime dimensión de datos)
    Retorna los datos de entrenamiento y de prueba con las imágenes (redimensionadas) de la carpeta.
    """
    # Crea lista con imágenes
    imagenes = []
    etiquetas = []
    
    for imagen in os.listdir(filepath): # Va archivo por archivo en la carpeta
        image = img.imread(os.path.join(filepath, imagen)) #Carga la imagen
        img_copy = image.copy() # Para conservar originales
        img_resize = cv2.resize(img_copy, dsize=(64, 64), interpolation=cv2.INTER_NEAREST) 
        imagenes.append(img_resize)
        etiquetas.append(name)

    #Crea y separa datos
    imagenes = np.asarray(imagenes)
    #etiquetas = np.asarray([etiquetas])
    
    # Hace un shuffle
    np.random.shuffle(imagenes)
    
    #Datos de entrenamiento
    x_train = imagenes[0:int(len(imagenes)*0.8)]
    y_train = np.asarray([etiquetas[0:int(len(etiquetas)*0.8)]]).T
    
    #Datos de prueba
    x_test  = imagenes[int(len(imagenes)*0.8):]
    y_test  = np.asarray([etiquetas[int(len(etiquetas)*0.8):]]).T
    
    if print_dim:
        print("Dimensiones\n")
        print("x_train : ", x_train.shape)
        print("y_train : ", y_train.shape)
        print()
        print("x_test  : ", x_test.shape)
        print("y_test  : ", y_test.shape)
    
    return x_train, y_train, x_test, y_test


# Captura imágenes del rostro para el entrenamiento
def capturas(name, num_img):
    """ Abre la cámara local"""
    
    web_cam = cv2.VideoCapture(0)

    cascPath = "face_recognitionOpenCv2-master/Cascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    count = 0

    while(True):
        _, imagen_marco = web_cam.read()

        rostro = faceCascade.detectMultiScale(imagen_marco, 1.5, 5)

        for(x,y,w,h) in rostro:
            cv2.rectangle(imagen_marco, (x,y), (x+w, y+h), (0,200,0), 4)
            count += 1

            cv2.imwrite("images/{}/{}_".format(name, name) + str(count) + ".jpg", imagen_marco[y:y+h, x:x+w])
            cv2.imshow("Creando Dataset", imagen_marco)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Detiene la cámara presionando 'q'
            break

        elif count >= num_img:  # Número de captura de imágenes
            break


    # Cuando todo está hecho, liberamos la captura
    web_cam.release()
    cv2.destroyAllWindows()
    

#Tomará una imagen de la persona después del entrenamiento para identificarla.
def img_predict(num_img=1):

    web_cam = cv2.VideoCapture(0)

    cascPath = "face_recognitionOpenCv2-master/Cascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    count = 0

    while(True):
        _, imagen_marco = web_cam.read()

        rostro = faceCascade.detectMultiScale(imagen_marco, 1.5, 5)

        for(x,y,w,h) in rostro:
            cv2.rectangle(imagen_marco, (x,y), (x+w, y+h), (0,200,0), 4)
            count += 1

            cv2.imwrite("predict/predict" + ".jpg", imagen_marco[y:y+h, x:x+w])
            cv2.imshow("Identificado", imagen_marco)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Detiene la cámara presionando 'q'
            break

        elif count >= num_img:  # Número de captura de imágenes
            break
            
    web_cam.release()
    cv2.destroyAllWindows()
    