# app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Compteur d'Objets Stable", layout="wide")
st.title("✅ Compteur d'Objets Stable")

# --- Barre Latérale avec le CHOIX DE LA STRATÉGIE ---
with st.sidebar:
    st.header("1. Stratégie d'Isolation")
    detection_method = st.radio(
        "Comment vos objets se distinguent-ils le mieux ?",
        ('Par Luminosité (fort contraste)', 'Par Couleur')
    )

    st.markdown("---")

    # --- Section des paramètres qui change en fonction de la stratégie ---
    if detection_method == 'Par Luminosité (fort contraste)':
        st.header("Paramètres de Luminosité")
        st.info("Idéal pour les objets sombres sur fond clair, ou vice-versa.")
        global_threshold = st.slider("Seuil de luminosité", 0, 255, 170)
        invert = st.checkbox("Inverser (objets sombres sur fond clair)", value=True)
    else: # Par Couleur
        st.header("Paramètres de Couleur (HSV)")
        st.info("Idéal quand la couleur est le seul différenciateur.")
        h_min = st.slider("Teinte Min", 0, 179, 0)
        h_max = st.slider("Teinte Max", 0, 179, 25)
        s_min = st.slider("Saturation Min", 0, 255, 100)
        s_max = st.slider("Saturation Max", 0, 255, 255)
        v_min = st.slider("Valeur (Luminosité) Min", 0, 255, 100)
        v_max = st.slider("Valeur (Luminosité) Max", 0, 255, 255)

    st.markdown("---")
    st.header("Paramètres de Comptage")
    st.info("Utilisés après l'isolation pour séparer et filtrer les objets.")
    dist_threshold = st.slider("Seuil de séparation", 0.01, 1.0, 0.2, 0.01)
    min_area = st.slider("Taille minimale de l'objet (px²)", 10, 5000, 100)

# --- Chargement de l'image ---
st.header("Étape 1 : Charger une image")
up = st.file_uploader("Déposez une image", type=["jpg", "jpeg", "png", "bmp", "tif"])

if not up:
    st.info("Veuillez charger une image pour commencer.")
    st.stop()

# --- Traitement de l'Image ---
image_pil = Image.open(up).convert("RGB")
image_bgr = np.array(image_pil)
image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# --- ÉTAPE 1: ISOLATION (selon la stratégie choisie) ---
mask = np.zeros(gray.shape, dtype="uint8")
if detection_method == 'Par Luminosité (fort contraste)':
    _, thresh = cv2.threshold(gray, global_threshold, 255, cv2.THRESH_BINARY)
    if invert:
        mask = cv2.bitwise_not(thresh)
    else:
        mask = thresh
else: # Par Couleur
    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

# --- ÉTAPE 2: NETTOYAGE DU MASQUE (commun aux deux stratégies) ---
kernel = np.ones((3,3), np.uint8)
# Enlever le bruit et les petites imperfections
cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

# --- ÉTAPE 3: SÉPARATION AVEC WATERSHED (commun aux deux stratégies) ---
sure_bg = cv2.dilate(cleaned_mask, kernel, iterations=3)
dist_transform = cv2.distanceTransform(cleaned_mask, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, dist_threshold * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
cv2.watershed(image_bgr, markers)

# --- ÉTAPE 4: EXTRACTION ET COMPTAGE (commun aux deux stratégies) ---
final_contours = []
for label in np.unique(markers):
    if label <= 1: continue
    label_mask = np.zeros(gray.shape, dtype="uint8")
    label_mask[markers == label] = 255
    contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    st.image(cleaned_mask, use_column_width='always')
    st.info("Le résultat de la stratégie choisie. Les objets doivent être blancs.")
with col3:
    st.subheader("Résultat Final")
    st.image(output, channels="BGR", use_column_width='always')
    
st.metric("Nombre d'objets détectés", len(particles))
