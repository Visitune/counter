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
    
    # Rendre les paramètres du code original interactifs
    blur_ksize = st.slider("Force du flou (impair)", 1, 15, 5, 2)
    min_area = st.slider("Taille minimale de l'objet (px²)", 1, 500, 10)
    
    # Ajouter le contrôle pour l'inversion
    st.header("2. Type d'Image")
    invert = st.checkbox("Mes objets sont sombres sur un fond clair", help="Cochez cette case si vos objets (particules) sont plus sombres que le fond.")

# --- Chargement de l'image ---
st.header("Étape 1 : Charger une image")
up = st.file_uploader("Déposez une image ici", type=["jpg", "jpeg", "png", "bmp", "tif"])

if not up:
    st.info("Veuillez charger une image pour commencer.")
    st.stop()

# --- Traitement de l'Image ---
# Convertir l'upload en image OpenCV
image_pil = Image.open(up).convert("RGB")
image = np.array(image_pil)
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Appliquer la logique du script original
gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Inverser si la case est cochée
if invert:
    thresh = cv2.bitwise_not(thresh)

# Trouver les contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrer par taille
particles = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# Dessiner les contours sur une copie de l'image
output = image_bgr.copy()
cv2.drawContours(output, particles, -1, (0, 255, 0), 2)

# --- Affichage des Résultats ---
st.header("Étape 2 : Analyser les résultats")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Image Originale")
    st.image(image_bgr, channels="BGR", use_column_width=True)

with col2:
    st.subheader("Masque Binaire")
    st.image(thresh, use_column_width=True)
    st.info("C'est ce que 'voit' l'ordinateur. Les objets à compter doivent être en blanc.")

with col3:
    st.subheader("Résultat Final")
    st.image(output, channels="BGR", use_column_width=True)
    
st.metric("Nombre d'objets détectés", len(particles))
