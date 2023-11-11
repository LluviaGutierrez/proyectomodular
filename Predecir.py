#PROYECTO MODULAR INBI CUCEI 2023-B
#TITULO PROYECTO: Machine Learning para el Análisis de Imágenes en la Identificación de Patrones Cancerígenos
#Participantes del equipo:
#-Angela Fernanda Palacios Gaxiola -Lluvia Mariana Gutierrez Interian -Edna Alejandra Larrinaga Garcia
#Asesor: Francisco Javier Alvarez Padilla

#Este codigo se utilizo para realizar las predicciones, tambien va incluido en la interfaz.

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import feature
from skimage.measure import find_contours
from sklearn.metrics import roc_auc_score
from skimage.segmentation import find_boundaries
from skimage.draw import set_color
from skimage.draw import rectangle_perimeter
import torch
import sys
import cv2

import tensorflow as tf
print(tf.__version__)

IMG_WIDTH = 1024
IMG_HEIGHT = 1024
IMG_CHANNELS = 3


# Cargar el modelo previamente entrenado
loaded_model = tf.keras.models.load_model(r'C:\Users\Lluvia\Downloads\trained_model_1024_1_funciona.keras',compile=False, safe_mode=False)


# Ruta de la imagen de prueba que deseas predecir
test_image_name = 'tumor_040_roi.tif'
test_image_path = r'C:\Users\Lluvia\Documents\UNIVERSIDAD\Proyecto Modulares\visualstudio\Codigos\Nuevas_tumor\tumor_040_roi.tif'# +  test_image_name

# Cargar la imagen de prueba y la máscara
test_img = imread(test_image_path)[:,:,:3]

def resize_images(img):
    IMG_WIDTH = 1024 #inicial
    IMG_HEIGHT = 1024  #inicial
    IMG_CHANNELS = 3
    resized_image = resize(img, (IMG_HEIGHT, IMG_WIDTH), anti_aliasing=False, mode='reflect')
    resized_image = (255 * resized_image).astype(np.uint8)
    return resized_image

def prediction(test_img):
    # Realizar la predicción
    global loaded_model
    resized_img = resize_images(test_img)
    resized_img = resized_img[np.newaxis, ...]
    prediction = loaded_model.predict(resized_img)
    # Binarizar la máscara de predicción y redimensionarla a la forma de la imagen original
    binary_prediction =  resize(prediction[0, ..., 0], (test_img.shape[0], test_img.shape[1]), mode='constant', preserve_range=True)
    # Asegurarse de que los valores estén en el rango adecuado
    binary_prediction = (binary_prediction - binary_prediction.min()) / (binary_prediction.max() - binary_prediction.min())
    binary_prediction = (binary_prediction > 0.5).astype(np.uint8)
    # Detectar contornos en la máscara binarizada
    contours = find_contours(binary_prediction, 0.5)
    # Crear una imagen en blanco para superponer las máscaras
    green_color = [0, 255, 0]
    yellow_color = [255, 255, 0]
    # Superponer la máscara en la imagen original
    pred = np.copy(test_img)
    pred[binary_prediction == 1] = [0, 255, 0]  # Resaltar en verde las áreas predichas como tumor en la imagen original
  
    overlay_pred = np.copy(test_img)
    for contour in contours:
        for point in contour:
            y, x = point
            rr, cc = rectangle_perimeter((y-1, x-1), extent=(3, 3), shape=overlay_pred.shape)  # Definir un rectángulo alrededor del punto para aumentar el grosor
            overlay_pred[rr, cc] = yellow_color  # Resaltar en amarillo los píxeles adyacentes para hacer los contornos más gruesos



    return overlay_pred, pred, binary_prediction

def process_data(image, img_id):
        IMG_WIDTH = 1000
        IMG_HEIGHT = 1000
        limit = 2000
        height, width, *channels = image.shape
        pred = np.zeros((height, width, 3), dtype=np.uint8)
        binary = np.zeros((height, width), dtype=np.uint8)
        contornos = np.zeros((height, width, 3), dtype=np.uint8)
        if height > limit:
          if width > limit:
            horizontal=int(width/IMG_WIDTH)
            vertical=int(height/IMG_HEIGHT)
            ancho= 0
            alto = 0
            final_a = 0
            final_h = 0
            parches = []
            max_value = 0
            print(f'Archivo {img_id} con medidas {image.shape} es vertical {vertical} y horizontal {horizontal}')
            for i in range(1, horizontal+1):
              ancho = final_a
              final_h = 0
              if i == horizontal:
                final_a = width
              else:
                final_a = int((width/horizontal)*i)
              for j in range(1,vertical+1):
                alto = final_h
                if j == vertical:
                  final_h = height
                else:
                  final_h = int((height/vertical)*j)
                parche = image[alto:final_h,ancho:final_a,:]
                overlay_pred, predic, binary_prediction = prediction(parche)
                binary[alto:final_h,ancho:final_a] = binary_prediction
                pred[alto:final_h,ancho:final_a,:] = predic
                contornos[alto:final_h,ancho:final_a,:] = overlay_pred
          else:
            if width > 500:
              horizontal=int(width/IMG_WIDTH)
              vertical=int(height/IMG_HEIGHT)
              print(f'Archivo {img_id} con medidas {image.shape} es vertical {vertical} y horizontal {horizontal}')
              ancho = 0
              final_a = width
              alto = 0
              final_h = 0
              parches = []
              for i in range(1,vertical+1):
                alto = final_h
                if i == vertical:
                  final_h = height
                else:
                  final_h=int((height/vertical)*i)
                parche = image[alto:final_h,ancho:final_a,:]
                overlay_pred, predic, binary_prediction = prediction(parche)
                binary[alto:final_h,ancho:final_a] = binary_prediction
                pred[alto:final_h,ancho:final_a,:] = predic
                contornos[alto:final_h,ancho:final_a,:] = overlay_pred
        else:
          if width>limit:
            horizontal=int(width/IMG_WIDTH)
            vertical=int(height/IMG_HEIGHT)
            print(f'Archivo {img_id} con medidas {image.shape} es vertical {vertical} y horizontal {horizontal}')
            ancho = 0
            alto = 0
            final_h = height
            final_a = 0
            parches=[]
            for i in range(1,horizontal+1):
              if i ==horizontal:
                final_a= width
              else:
                final_a=int((height/horizontal)*i)
              parche = image[alto:final_h,ancho:final_a,:]
              overlay_pred, predic, binary_prediction = prediction(parche)
              binary[alto:final_h,ancho:final_a] = binary_prediction
              pred[alto:final_h,ancho:final_a,:] = predic
              contornos[alto:final_h,ancho:final_a,:] = overlay_pred
          else:
           if 700 <= height <= limit and 700 <= width <= limit:
              print(f'Todo nice aqui {img_id} con {image.shape}')
              overlay_pred, predic, binary_prediction = prediction(image)
              binary = binary_prediction
              pred = predic
              contornos = overlay_pred
           else:
             if 400 < height < 1024 and 400 < width < 1024:
              horizontal=int(width/IMG_WIDTH)
              vertical=int(height/IMG_HEIGHT)
              IMG_HEIGHT = 1024
              IMG_WIDTH = 1024
              print(f'ENTRE 400 Y 1024: Archivo {img_id} con medidas {image.shape} es vertical {vertical} y horizontal {horizontal}')
              padded_image = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
              padded_image[:height, :width, :] = image
              overlay_pred, predic, binary_prediction = prediction(padded_image)
              binary = binary_prediction[:height,:width]
              pred = predic[:height,:width,:]
              contornos = overlay_pred[:height,:width,:]
             else:
              if height > 1024:
                IMG_HEIGHT = 1024
                IMG_WIDTH = 1024
                horizontal=int(width/IMG_WIDTH)
                vertical=int(height/IMG_HEIGHT)
                padded_image = np.zeros((height, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
                padded_image[:height, :width, :] = image
                overlay_pred, predic, binary_prediction = prediction(padded_image)
                binary = binary_prediction[:height,:width]
                pred = predic[:height,:width,:]
                contornos = overlay_pred[:height,:width,:]
                print(f'HEIGHT >1024: Archivo {img_id} con medidas {image.shape} es vertical {vertical} y horizontal {horizontal}')
              elif width >1024:
                IMG_HEIGHT = 1024
                IMG_WIDTH = 1024
                horizontal=int(width/IMG_WIDTH)
                vertical=int(height/IMG_HEIGHT)
                padded_image = np.zeros((IMG_HEIGHT, width, IMG_CHANNELS), dtype=np.uint8)
                padded_image[:height, :width, :] = image
                overlay_pred, predic, binary_prediction = prediction(padded_image)
                binary = binary_prediction[:height,:width]
                pred = predic[:height,:width,:]
                contornos = overlay_pred[:height,:width,:]
                print(f'WIDTH > 1024: Archivo {img_id} con medidas {image.shape} es vertical {vertical} y horizontal {horizontal}')
              else:
                print(f'No es posible utilizar {img_id} debido a sus dimensiones {image.shape}')

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(contornos)
        axes[0].set_title('Predicciones con contornos resaltados')
        axes[0].axis('off')
        axes[1].imshow(binary)
        axes[1].set_title('Máscara predicha con contornos resaltados')
        axes[1].axis('off')
        plt.show()


process_data(test_img, test_image_name)