#Importar para TKinter
import tkinter
from tkinter import Canvas, filedialog, messagebox, font, PhotoImage, Label
from PIL import Image, ImageTk
#Importar para Prediccion
import numpy as np
import tensorflow as tf
from skimage.io import imread
from skimage.transform import resize
import os

#Importar para Predecir
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

# Variables globales para almacenar ruta de modelo entrenado y ruta de imagen ingresada
ruta_archivo = None
ruta_modelo = None
nombre_archivo = None
loaded_model = None
test_img = None
test_image_name = None

#FUNCIONES
def cargar_imagen():
    global ruta_archivo, canvas #globales para que sean return
    ruta_archivo = filedialog.askopenfilename(filetypes = [("Archivos TIFF", "*.tiff *.tif")]) #abre seleccion de archivo
    if ruta_archivo:
        img = Image.open(ruta_archivo)
        img = img.resize((560,500))
        img = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor = "nw", image = img) #se crea que pueda usarse en canva
        canvas.image = img
        nombre_archivo = os.path.basename(ruta_archivo)
        muestra = tkinter.Label(ventana2, width = 40, height = 2, text = f"Nombre del archivo: {nombre_archivo}", font = 15, bg = "#CD8C95")
        muestra.place(x = 820, y = 160)
    else: messagebox.showinfo(message = "Seleccion Cancelada", title = "Cancel")

def starts():
    global ventana, ventana2, canvas
    ventana2 = tkinter.Toplevel(ventana) #abrir ventana secundaria
    ventana2.title("MACHINE LEARNING PARA EL ANÁLISIS DE IMÁGENES EN LA IDENTIFICACIÓN DE PATRONES CANCERÍGENOS")
    ventana2.geometry("1366x768")
    ventana2.configure(bg = "#EEE0E5") 
    style = font.Font(weight = "bold", size = 35, family = "Times New Roman")
    titulo = Label(ventana2, text = "Ingresar Imagen Para Analizar", font = (style))
    titulo.config(fg = "#8B0A50", bg = "#EEE0E5")
    titulo.pack(padx = 0, pady = 20)
    back = tkinter.Button(ventana2, text = "Regresar a Menú Principal", width = 25, height = 3, bg = "#8B0A50", fg = "#FFFFFF", font = 15, borderwidth = 5, command = vent) #crear boton de regresar a principal
    back.place(x = 1110, y = 610)
    help = tkinter.Button(ventana2, text = "Ayuda", width = 15, height = 2, bg = "#8B0A50", fg = "#FFFFFF", command = ayuda2)
    help.place(x = 0, y = 0)
    ty = font.Font(weight = "bold", size = 20, family = "Times New Roman")
    i1 = tkinter.Label(ventana2, text = "Muestra Ingresada", width = 35, bg = "#CD8C95", font = (ty))
    i1.place(x = 170, y = 100)
    imagen = tkinter.Frame(ventana2, width = 560, height = 500) #crear un frame para poner imagen a analizar
    imagen.place(x = 170, y = 160)
    canvas = tkinter.Canvas(imagen, width = 560, height = 500, bg = "#EEE0E5") #crear canva dentro del frame para poner la imagen
    canvas.pack()
    ty1 = font.Font(size = 16, family = "Cascadia Mono")
    cargar = tkinter.Button(ventana2, text = "Cargar Imagen", width = 20, height = 3, bg = "#8B0A50", fg = "#FFFFFF", borderwidth = 5, font = (ty1), command = cargar_imagen) #crear boton de cargar
    cargar.place(x = 870, y = 250)
    procesar = tkinter.Button(ventana2, text = "Procesar", width = 20, height = 3, bg = "#F5DEB3", borderwidth = 5, font = (ty1), command = proces) #crear boton de analizar
    procesar.place(x = 870, y = 425)
    ventana.withdraw() #quitar ventana principal

def proces():
    global ventana, ventana2, ventana3
    ventana3 = tkinter.Toplevel(ventana) #abrir ventana nueva
    ventana3.title("MACHINE LEARNING PARA EL ANÁLISIS DE IMÁGENES EN LA IDENTIFICACIÓN DE PATRONES CANCERÍGENOS")
    ventana3.geometry("1366x768") 
    ventana3.configure(bg = "#EEE0E5") 
    ventana2.withdraw() #quitar ventana de cargar
    style = font.Font(weight = "bold", size = 35, family = "Times New Roman")
    titulo = Label(ventana3, text = "Etapa de Predicción", font = (style))
    titulo.config(fg = "#8B0A50", bg = "#EEE0E5")
    titulo.pack(padx = 0, pady = 20)
    back = tkinter.Button(ventana3, text = "Regresar a Menú Principal", width = 25, height = 3, bg = "#8B0A50", fg = "#FFFFFF", borderwidth = 5, font = 15, command = vent) #crear boton de regresar a principal
    back.place(x = 1110, y = 610)
    help = tkinter.Button(ventana3, text = "Ayuda", width = 15, height = 2, bg = "#8B0A50", fg = "#FFFFFF", command = ayuda3)
    help.place(x = 0, y = 0)
    ty1 = font.Font(size = 16, family = "Cascadia Mono")
    direct = tkinter.Button(ventana3, text = "Cargar Modelo", width = 20, height = 3, bg = "#8B668B", fg = "#FFFFFF", borderwidth = 5, command = modelo, font = (ty1)) #crear boton de cargar modelo
    direct.place(x = 100, y = 150)
    analize = tkinter.Button(ventana3, text = "Comenzar Predicción", width = 20, height = 3, bg = "#CD9B9B", fg = "#FFFFFF", borderwidth = 5, font = (ty1), command = predict) #crear boton de analizar
    analize.place(x = 550, y = 150)
    vista = tkinter.Button(ventana3, text = "Cambiar Vista", width = 20, height = 3, bg = "#8B668B", fg = "#FFFFFF", borderwidth = 5, font = (ty1)) #crear boton de cargar modelo
    vista.place(x = 1000, y = 150)
    ty = font.Font(weight = "bold", size = 20, family = "Times New Roman")
    p = tkinter.Label(ventana3, text = "Datos de Predicción", width = 30, bg = "#CD8C95", font = (ty))
    p.place(x = 430, y = 310)
    p2= tkinter.Label(ventana3, text = "Avance: ", width = 10, bg = "#CD8C95", font = (ty))
    p2.place(x = 430, y = 400)
    p3 = tkinter.Label(ventana3, width = 28, bg = "#CD8C95", font = (ty))
    p3.place(x = 400, y = 500)
    #datos = tkinter.Frame(ventana3, width = 500, height = 300, bg = "#EEE0E5") #crear un frame para poner imagen a analizar
    #datos.place(x = 430, y = 380)

def modelo():
    global ruta_modelo
    ruta_modelo = filedialog.askopenfilename(filetypes = [("Archivos .KERAS", "*.keras")]) #abre seleccion de archivo modelo

def predict():
  import predecirplis
  
def vent():
    global ventana, ventana2, ventana3
    ventana.deiconify() #reaparecer ventana principal
    ventana2.destroy() #destruir ventana secundaria
    ventana3.destroy() #destruir ventana de analizar
    
def ayuda():
    messagebox.showinfo(message = "Antes de utilizar el programa debe tener preparado lo siguiente:\n"
                        +">>Una imagen a analizar de Biopsia del Nodo Linfático Centinela (SLNB) en formato .tif\n"
                        +">>El modelo de entrenamiento descargado y guardado en su lugar de preferencia\n\n"
                        +"Para comenzar haga clic en el botón con el logo del proyecto", title = "INSTRUCCIONES")

def ayuda2():
    messagebox.showinfo(message = "--> Ingresar una imagen <--\n"
                        +">>Imagen con extensión .tif\n"
                        +">>Dar click en <CARGAR>. La imagen cargada será mostrada en pantalla\n"
                        +">>Posteriormente dar click en <PROCESAR> para dirigirse a la siguiente pantalla\n\n"
                        +"Para volver al menú principal presiona <REGRESAR>", title = "INSTRUCCIONES")
    
def ayuda3():
    messagebox.showinfo(message = "--> Predicción <--\n"
                        +">>Primero debe ingresar la dirección de acceso del modelo descargado en el botón <CARGAR MODELO>\n"
                        +">>Para comenzar la predicción dar click en <COMENZAR PREDICCIÓN>"
                        +">>Se muestra la predicción realizada de dos maneras en ventanas emergentes:\n"
                        +"   >>Máscara binaria creada\n"
                        +"   >>Máscara ubicada dentro de la imagen de tejido\n"
                        +">>Para cambiar la vista a solo contorno dar click a <CAMBIAR VISTA>\n\n"
                        +"Para volver al menú principal presiona <REGRESAR>", title = "INSTRUCCIONES")

def infos():
    messagebox.showinfo(message = "Realizado por las estudiantes de Ingeniería Biomédica:\n"
                        +"\n"
                        + "                     Palacios Gaxiola Angela Fernanda\n"
                        + "                     Gutiérrez Interián LLuvia Mariana\n"
                        + "                     Larrinaga Garcia Edna Alejandra\n\n"
                        + "Este es un prototipo de inteligencia Machine Learning para la detección de células cancerígenas en imágenes captadas durante la técnica Biopsia del Nodo Linfático Centinela (SLNB) ", title = "INFORMACIÓN")


ventana = tkinter.Tk() #crear ventana que va a contener todos los objetos graficos
ventana.title("MACHINE LEARNING PARA EL ANÁLISIS DE IMÁGENES EN LA IDENTIFICACIÓN DE PATRONES CANCERÍGENOS") #titulo de ventana
ventana.configure(bg = "#FFFFFF") 
ventana.geometry("1366x768") #tamaño ventana
#Cargar imagenes de fondo
imgfondo = PhotoImage(file = r"C:\Users\Lluvia\Documents\UNIVERSIDAD\Proyecto Modulares\visualstudio\interfaz\1.png")
btn = PhotoImage(file = r"C:\Users\Lluvia\Documents\UNIVERSIDAD\Proyecto Modulares\visualstudio\interfaz\2.png")
udg = PhotoImage(file = r"C:\Users\Lluvia\Documents\UNIVERSIDAD\Proyecto Modulares\visualstudio\interfaz\3.png")
cucei = PhotoImage(file = r"C:\Users\Lluvia\Documents\UNIVERSIDAD\Proyecto Modulares\visualstudio\interfaz\4.png")
icono = PhotoImage(file = r"C:\Users\Lluvia\Documents\UNIVERSIDAD\Proyecto Modulares\visualstudio\interfaz\5.png")
ventana.iconphoto(True, icono) #icono de ventana
fondo = Label(ventana, width = 1400, height = 1000, image = imgfondo, bg = "#FFFFFF") #etiqueta para meter el fondo
fondo.pack()
f = Label(ventana, image = udg, bg = "#FFFFFF") 
f.place(x = 0, y = 430)
f2 = Label(ventana, image = cucei, bg = "#FFFFFF") 
f2.place(x = 1180, y = 428)
#TITULO
style = font.Font(weight = "bold", size = 32, family = "Times New Roman") #estilo
titulo = Label(ventana, text = "Machine Learning Para El Análisis De Imágenes En\nIdentificación De Patrones Cancerígenos", font = (style)) #texto
titulo.config(fg = "#C71585", bg = "#FFFFFF")
titulo.place(x = 240, y = 0)
#BOTONES
start = tkinter.Button(ventana, width = 250, height = 150, command = starts, image = btn, bg = "#FFFFFF") #crear boton de comenzar
start.place(x = 555, y = 269)
info = tkinter.Button(ventana, text = "Ver más...", width = 10, height = 2, bg = "#8B0A50", fg = "#FFFFFF", command = infos) #crear boton de informacion
info.place(x = 0, y = 0)
help = tkinter.Button(ventana, text = "Ayuda", width = 10, height = 2, bg = "#8B0A50", fg = "#FFFFFF", command = ayuda) #crear boton de ayuda
help.place(x = 80, y = 0)
close = tkinter.Button(ventana, text = "Salir", width = 13, height = 2, bg = "#8B0A50", fg= "#FFFFFF", borderwidth = 5, font = 15, command = ventana.destroy)
close.place(x = 1233, y = 0)

ventana.mainloop() ##lleva registro de todo lo que esta sucediendo en la ventana, hace que salga la pantalla


