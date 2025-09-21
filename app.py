# app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Compteur d'Objets Avancé", layout="wide")
st.title("🔬 Compteur d'Objets Avancé (avec Watershed)")

# --- Barre Latérale avec les Contrôles ---
with st.sidebar:
    st.header("1. Paramètres de Binarisation")
    st.info("Ajustez ce curseur pour que les objets apparaissent en blanc sur le masque, même s'ils se touchent.")
    threshold_level = st.slider("Seuil manuel", 0, 255, 150)
    invert = st.checkbox("Inverser le masque", help="Cochez si vos objets apparaissent en noir au lieu de blanc.")

    st.header("2. Paramètres de Comptage")
    min_area = st.slider("Taille minimale de l'objet (px²)", 10, 5000, 100)
    use_watershed = st.checkbox("✅ Activer la séparation Watershed", value=True, help="Algorithme puissant pour séparer les objets qui se touchent.")
    
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

# Binarisation manuelle, beaucoup plus fiable
_, thresh = cv2.threshold(gray, threshold_level, 255, cv2.THRESH_BINARY)
if invert:
    thresh = cv2.bitwise_not(thresh)

# --- LOGIQUE DE COMPTAGE ---
final_contours = []
if use_watershed:
    # --- Algorithme Watershed pour séparer les objets ---
    # 1. Trouver l'arrière-plan certain
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(thresh, kernel, iterations=3)

    # 2. Trouver le premier-plan certain
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 3. Trouver la région inconnue
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 4. Créer des marqueurs pour les composants
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Pour que l'arrière-plan ne soit pas 0, mais 1
    markers[unknown == 255] = 0 # Marquer la région inconnue avec 0

    # 5. Appliquer l'algorithme Watershed
    cv2.watershed(image_bgr, markers)

    # 6. Extraire les contours de chaque objet séparé
    unique_labels = np.unique(markers)
    for label in unique_labels:
        if label <= 1:  # Ignorer l'arrière-plan et les bordures
            continue
        
        # Créer un masque pour le label actuel et trouver son contour
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == label] = 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_contours.extend(contours)
else:
    # Méthode simple (qui ne sépare pas les objets)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_contours = contours

# Filtrer les contours finaux par taille
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
    st.subheader("Masque Binaire")
    st.image(thresh, use_column_width='always')
    st.info("Ajustez le 'Seuil manuel' pour que les objets soient bien blancs.")

with col3:
    st.subheader("Résultat Final")
    st.image(output, channels="BGR", use_column_width='always')
    
st.metric("Nombre d'objets détectés", len(particles))
