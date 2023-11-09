#PROYECTO MODULAR INBI CUCEI 2023-B
#TITULO PROYECTO: Machine Learning para el Análisis de Imágenes en la Identificación de Patrones Cancerígenos
#Participantes del equipo:
#-Angela Fernanda Palacios Gaxiola -Lluvia Mariana Gutierrez Interian -Edna Alejandra Larrinaga Garcia
#Asesor: Francisco Javier Alvarez Padilla

#Este codigo se utilizo para crear las mascaras de las imagenes originales, tanto test como train.

import sys
for p in sys.path:
    print (p)

import os
#Direccion a bin
bin_directory = r"C:\Users\Lluvia\Downloads\Computer Stuff\ASAP 2.2\bin"
#Añadir direccion de bin a python
sys.path.append(bin_directory)
for p in sys.path:
    print (p)

#OPENSLIDE
from openslide import OpenSlide
import openslide
#Ruta a la carpeta de binarios de OpenSlide
openslide_path = r'C:\Users\Lluvia\Downloads\Computer Stuff\openslide-win64-20230414\openslide-win64\bin'
#Agrega la ruta a la variable de entorno PATH
os.environ['PATH'] = openslide_path + ';' + os.environ['PATH']
#Verifica que la ruta se haya agregado correctamente
for p in sys.path:
    print (p)

import numpy as np
# Import multiresolutionimageinterface
import multiresolutionimageinterface as mir

#Carga las imagenes en la carpeta seleccionada
def cargar_imagen(ruta_imagen):
    reader = mir.MultiResolutionImageReader()
    return reader.open(ruta_imagen)

#Carga anotaciones respectivas
def cargar_anotaciones(ruta_xml):
    annotation_list = mir.AnnotationList()
    xml_repository = mir.XmlRepository(annotation_list)
    xml_repository.setSource(ruta_xml)
    xml_repository.load()
    return annotation_list

#Crear la mascara binaria
def crear_mascara(ruta_imagen, ruta_mascara, ruta_xml, label_map, conversion_order):
    mr_image = cargar_imagen(ruta_imagen)
    annotation_list = cargar_anotaciones(ruta_xml)
    annotation_mask = mir.AnnotationToMask()
    annotation_mask.convert(annotation_list, ruta_mascara, mr_image.getDimensions(), mr_image.getSpacing(), label_map, conversion_order)

#Seleccionar carpeta con imagenes originales
def procesar_carpeta_imagenes(carpeta_imagenes, carpeta_anotaciones, carpeta_mascaras, extensiones_validas, label_map, conversion_order):
    for nombre_archivo in os.listdir(carpeta_imagenes):
        if any(nombre_archivo.endswith(ext) for ext in extensiones_validas):
            ruta_imagen = os.path.join(carpeta_imagenes, nombre_archivo)
            ruta_xml = os.path.join(carpeta_anotaciones, os.path.splitext(nombre_archivo)[0] + '.xml')
            ruta_mascara = os.path.join(carpeta_mascaras, os.path.splitext(nombre_archivo)[0] + '_mask.tif')
            crear_mascara(ruta_imagen, ruta_mascara, ruta_xml, label_map, conversion_order)

# Configuración
carpeta_imagenes = r'C:\Users\Lluvia\Documents\UNIVERSIDAD\Proyecto Modulares\visualstudio\n'
carpeta_anotaciones = r'C:\Users\Lluvia\Documents\UNIVERSIDAD\Proyecto Modulares\visualstudio\n2'
carpeta_mascaras = r'C:\Users\Lluvia\Documents\UNIVERSIDAD\Proyecto Modulares\visualstudio\n3'
extensiones_validas = ['.tif', '.tiff', '.jpg', '.jpeg', '.png'] #tipo de imagenes aceptadas
camelyon17_type_mask = False
#Clasificacion de tejido segun anotaciones
label_map = {'metastases': 1, 'normal': 0} if camelyon17_type_mask else {'_0': 1, '_1': 1, '_2': 0}
conversion_order = ['metastases', 'normal'] if camelyon17_type_mask else  ['_0', '_1', '_2']

# Procesar la carpeta de imágenes
procesar_carpeta_imagenes(carpeta_imagenes, carpeta_anotaciones, carpeta_mascaras, extensiones_validas, label_map, conversion_order)
