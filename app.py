# app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Compteur d'Objets Avancé", layout="wide")
st.title("🔬 Compteur d'Objets Avancé")

# --- Barre Latérale avec les Contrôles ---
with st.sidebar:
    st.header("1. Paramètres de Détection")
    st.info("Utilisez ces paramètres pour isoler les objets.")
    block_size = st.slider("Taille du voisinage (impair)", 11, 255, 115, 2)
    C = st.slider("Sensibilité locale (C)", -20, 20, 2)
    
    st.header("2. Paramètres de Séparation")
    st.info("Ajustez ce seuil pour bien séparer les objets.")
    watershed_threshold = st.slider("Seuil de séparation", 0.1, 1.0, 0.5, 0.05)
    min_area = st.slider("Taille minimale de l'objet (px²)", 10, 5000, 150)
    
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

# --- ÉTAPE 1: SEUIL ADAPTATIF POUR ISOLER LES OBJETS ---
# Crée un masque binaire en se basant sur la luminosité locale
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, block_size, C)

# --- ÉTAPE 2: NETTOYAGE DU MASQUE ---
# Enlever le bruit
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# --- ÉTAPE 3: SÉPARATION AVEC WATERSHED ---
# Arrière-plan certain
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Premier-plan certain (les "coeurs" des objets)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, watershed_threshold * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# Région inconnue
unknown = cv2.subtract(sure_bg, sure_fg)

# Créer les marqueurs
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# Appliquer Watershed
cv2.watershed(image_bgr, markers)

# --- ÉTAPE 4: EXTRACTION ET COMPTAGE ---
final_contours = []
unique_labels = np.unique(markers)
for label in unique_labels:
    if label <= 1: continue # Ignorer fond et bordures
    
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[markers == label] = 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_contours.extend(contours)

particles = [cnt for cnt in final_contours if cv2.contourArea(cnt) > min_area]

output = image_bgr.copy()
cv2.drawContours(output, particles, -1, (0, 255, 0), 2)

# --- Affichage des Résultats ---
st.header("Étape 2 : Analyser les résultats")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Image Originale")
    st.image(image_bgr, channels="BGR", use_column_width='always')

with col2:
    st.subheader("Masque d'Isolation")
    st.image(opening, use_column_width='always')
    st.info("Les objets doivent être des formes blanches distinctes.")

with col3:
    st.subheader("Résultat Final")
    st.image(output, channels="BGR", use_column_width='always')
    
st.metric("Nombre d'objets détectés", len(particles))
