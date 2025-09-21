# app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Compteur d'Objets Interactif", layout="wide")
st.title("🔬 Compteur d'Objets Interactif")

# --- Barre Latérale avec les Contrôles ---
with st.sidebar:
    st.header("1. Paramètres de Détection")
    blur_ksize = st.slider("Force du flou (impair)", 1, 15, 3, 2)
    min_area = st.slider("Taille minimale de l'objet (px²)", 1, 1000, 50)
    
    st.header("2. Outil de Séparation")
    st.info("Utilisez l'érosion pour séparer les objets qui se touchent.")
    erosion_iterations = st.slider("Itérations d'érosion", 0, 10, 1)
    dilation_iterations = st.slider("Itérations de dilatation", 0, 10, 1)

    st.header("3. Type d'Image")
    invert = st.checkbox("Mes objets sont sombres sur un fond clair", help="Cochez cette case si vos objets (particules) sont plus sombres que le fond.")

# --- Chargement de l'image ---
st.header("Étape 1 : Charger une image")
up = st.file_uploader("Déposez une image ici", type=["jpg", "jpeg", "png", "bmp", "tif"])

if not up:
    st.info("Veuillez charger une image pour commencer.")
    st.stop()

# --- Traitement de l'Image ---
image_pil = Image.open(up).convert("RGB")
image_bgr = np.array(image_pil)
image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)

gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

if not invert: # L'inversion est maintenant logique : on veut que les objets soient blancs.
    thresh = cv2.bitwise_not(thresh)

# --- NOUVELLE ÉTAPE DE SÉPARATION ---
# Créer un "noyau" pour les opérations
kernel = np.ones((3, 3), np.uint8)
# Appliquer l'érosion pour "grignoter" les bords et séparer les objets
eroded_mask = cv2.erode(thresh, kernel, iterations=erosion_iterations)
# Appliquer la dilatation pour restaurer la taille des objets séparés
final_mask = cv2.dilate(eroded_mask, kernel, iterations=dilation_iterations)

# Trouver les contours sur le masque final, après séparation
contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

particles = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

output = image_bgr.copy()
cv2.drawContours(output, particles, -1, (0, 255, 0), 2)

# --- Affichage des Résultats ---
st.header("Étape 2 : Analyser les résultats")

# Ajouter une colonne pour visualiser l'étape de séparation
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Image Originale")
    st.image(image_bgr, channels="BGR", use_column_width='always')

with col2:
    st.subheader("Masque Binaire Initial")
    st.image(thresh, use_column_width='always')

with col3:
    st.subheader("Séparation (Érosion)")
    st.image(final_mask, use_column_width='always')
    st.info("C'est le masque après séparation. Les amas devraient être cassés.")

with col4:
    st.subheader("Résultat Final")
    st.image(output, channels="BGR", use_column_width='always')
    
st.metric("Nombre d'objets détectés", len(particles))
